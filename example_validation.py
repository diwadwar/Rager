import numpy as np
from aeon.transformations.collection.convolution_based import Rager
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import time

def rocketValidate(dataset='PSL extended dataset\\PSL_extended.npz',alignLengthMethod='padding',time_measure=False,
                   repetitions=30,verbose=True,n_kernels=None,max_dilations=None):
    # --- reading data ---
    dataset = np.load(,allow_pickle=True)

    data = dataset['data']
    labels = dataset['labels']

    # train-test split
    x_train_raw = data[0::2]
    y_train_raw = labels[0::2]

    x_test_raw = data[1::2]
    y_test_raw = labels[1::2]

    x_train_list = [series for subset in x_train_raw for series in subset]
    x_test_list = [series for subset in x_test_raw for series in subset]

    y_train = np.concatenate(y_train_raw)
    y_test = np.concatenate(y_test_raw)

    # max_len_train = max sample length in training subset or 10
    max_len_train = max(max(s.shape[-1] for s in x_train_list),10)

    # --- data alignment ---
    if alignLengthMethod.lower() == 'padding':
        # zero padding
        def pad_sequences(X_list,target_len,pad_value=0.0):
            result = []
            for series in X_list:
                current_len = series.shape[-1]
                if current_len < target_len:
                    pad_width = target_len - current_len
                    result.append(
                        np.pad(series,((0,0),(0,pad_width)),
                               constant_values=pad_value)
                    )
                elif current_len > target_len:
                    x_old = np.linspace(0,1,current_len)
                    x_new = np.linspace(0,1,target_len)
                    resampled = np.array([
                        np.interp(x_new,x_old,channel)
                        for channel in series
                    ])
                    result.append(resampled)
                else:
                    result.append(series)
            return np.stack(result)

        x_train = pad_sequences(x_train_list,max_len_train)
        x_test = pad_sequences(x_test_list,max_len_train)

    elif alignLengthMethod.lower() == 'interpolation':
        # time normalization by interpolation
        def resample_sequences(X_list,target_len):
            result = []
            for series in X_list:
                current_len = series.shape[-1]
                if current_len != target_len:
                    x_old = np.linspace(0,1,current_len)
                    x_new = np.linspace(0,1,target_len)
                    resampled = np.array([
                        np.interp(x_new,x_old,channel)
                        for channel in series
                    ])
                    result.append(resampled)
                else:
                    result.append(series)
            return np.stack(result)

        x_train = resample_sequences(x_train_list,max_len_train)
        x_test = resample_sequences(x_test_list,max_len_train)
    else:
        raise Exception("Unknown align length method.")

    # --- training and classification ---
    random_states = list(range(repetitions))  # [0, 1, 2, ..., repetitions-1]

    accuracies = []
    balanced_accuracies = []
    f1_scores = []

    train_time = 0
    test_time = 0

    for rs in random_states:
        # arguments for Rager
        kwargs = {'random_state':rs}
        if n_kernels is not None:
            kwargs['n_kernels'] = n_kernels
        if max_dilations is not None:
            kwargs['max_dilations_per_kernel'] = max_dilations

        clf = make_pipeline(
            Rager(**kwargs),
            StandardScaler(),
            RidgeClassifierCV()
        )

        if time_measure:
            # training
            start_train = time.perf_counter()
            clf.fit(x_train,y_train)
            train_time += time.perf_counter() - start_train

            # classification
            start_test = time.perf_counter()
            y_pred = clf.predict(x_test)
            test_time += time.perf_counter() - start_test
        else:
            clf.fit(x_train,y_train) # training
            y_pred = clf.predict(x_test) # classification

        # metrics
        acc = accuracy_score(y_test,y_pred)
        bal_acc = balanced_accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred,average='macro')

        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)
        f1_scores.append(f1)

        if verbose:
            print(f"random_state={rs:2d} | acc={acc:.4f} | bal_acc={bal_acc:.4f} | f1={f1:.4f}")

    # --- calculate and return statistics ---
    # conversion to numpy array
    accuracies = np.array(accuracies)
    balanced_accuracies = np.array(balanced_accuracies)
    f1_scores = np.array(f1_scores)

    if verbose:
        print("RESULTS SUMMARY")
        print(
            f"Mean accuracy: {accuracies.mean():.4f} | Standard deviation: {accuracies.std():.4f} | Min / Max: {accuracies.min():.4f} / {accuracies.max():.4f}")
        print(
            f"Mean balanced accuracy: {balanced_accuracies.mean():.4f}")
        print(
            f"Mean F1-score: {f1_scores.mean():.4f}")

    if time_measure:
        return (accuracies,accuracies.mean(),
                train_time,test_time,
                balanced_accuracies.mean(),f1_scores.mean())
    else:
        return (accuracies,accuracies.mean(),
                balanced_accuracies.mean(),f1_scores.mean())

# run validation
accuracies, meanAccuracies, meanBalAccuracies, meanF1Score = rocketValidate()






