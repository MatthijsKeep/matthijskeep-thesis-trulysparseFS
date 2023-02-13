            # if i == 0 or i == 1 or i == 5 or i == 10 or i % 10 == 0:
            #     start_time = time.time()
            #     print(f"The shape of the input layer weights is: {set_mlp.w[1].shape}")
            #     selected_features, importances = select_input_neurons(set_mlp.w[1], args.K)
            #     # print(set_mlp.w[1])
                
                
            #     # Print how many neurons in the input layer have a connection
            #     print(f"The number of neurons in the input layer with a connection is: {np.count_nonzero(set_mlp.w[1].sum(axis=1))}")
            #     # Print which neurons have a connection

            #     # print(f" The selected features are {selected_features}")
            #     selected_features_for_eval = pd.DataFrame(selected_features) # TODO: remove this and instead use the importances to select the features
            #     # convert to csv
            #     selected_features_for_eval = selected_features_for_eval.to_csv(header=False, index=False)

            #     selected_features_path = f"features/selected_features_{str(args.data)}_{i}_{str(args.model)}.csv"
            #     if not os.path.exists(os.path.dirname(selected_features_path)):
            #         os.makedirs(os.path.dirname(selected_features_path))

            #     with open(selected_features_path, 'w') as f:
            #         f.write(selected_features_for_eval)

            #     importances_for_eval = pd.DataFrame(importances)
            #     # convert to csv
            #     importances_for_eval = importances_for_eval.to_csv(header=False, index=False)

            #     importances_path = f"importances/importances_{str(args.data)}_{i}_{str(args.model)}.csv"
            #     if not os.path.exists(os.path.dirname(importances_path)):
            #         os.makedirs(os.path.dirname(importances_path))

            #     with open(importances_path, 'w') as f:
            #         f.write(importances_for_eval)
                
                
            #     print(f"The choosing and logging  of the {args.K} most important weights took {time.time() - start_time} seconds before the {i} epoch")
            
print(f"The shape of the input layer weights is: {set_mlp.w[1].shape}")
        selected_features, importances = select_input_neurons(set_mlp.w[1], k)
        # print(set_mlp.w[1])
        print(f"The choosing of the {k} most important weights took {time.time() - start_time} seconds")
        
        # Print how many neurons in the input layer have a connection
        print(f"The number of neurons in the input layer with a connection is: {np.count_nonzero(set_mlp.w[1].sum(axis=1))}")
        # Print which neurons have a connection

        print(f" The selected features are {selected_features}")
        selected_features_for_eval = pd.DataFrame(selected_features)
        # convert to csv
        selected_features_for_eval = selected_features_for_eval.to_csv(header=False, index=False)

        selected_features_path = f"features/selected_features_{str(args.data)}_{no_training_epochs}_{str(args.model)}.csv"
        if not os.path.exists(os.path.dirname(selected_features_path)):
            os.makedirs(os.path.dirname(selected_features_path))

        with open(selected_features_path, 'w') as f:
            f.write(selected_features_for_eval)

        importances_for_eval = pd.DataFrame(importances)
        # convert to csv
        importances_for_eval = importances_for_eval.to_csv(header=False, index=False)

        importances_path = f"importances/importances_{str(args.data)}_{no_training_epochs}_{str(args.model)}.csv"
        if not os.path.exists(os.path.dirname(importances_path)):
            os.makedirs(os.path.dirname(importances_path))

        with open(importances_path, 'w') as f:
            f.write(importances_for_eval)
        # take the selected features from the first column of the importances
        selected_features = importances[:, 0].astype(int)

        # change x_train to only have the selected features
        # reshape x_train_new and x_test to be 2D
        print(f"The shape of x_train is: {x_train.shape}")
        print(f"The shape of x_test is: {x_test.shape}")

        # change x_train and x_test to only have the selected features
        x_train_new = np.squeeze(x_train[:, selected_features])
        x_test_new = np.squeeze(x_test[:, selected_features])
        # change y_train and y_test from one-hot to single label
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # print all shapes
        print(f"The shape of x_train is: {x_train.shape}")
        print(f"The shape of x_train_new is: {x_train_new.shape}")
        print(f"The shape of x_test is: {x_test.shape}")
        print(f"The shape of x_test_new is: {x_test_new.shape}")
        print(f"The shape of y_train is: {y_train.shape}")
        print(f"The shape of y_test is: {y_test.shape}")

        # time the tesitng
        start_time = time.time()
        accuracy_topk = svm_test(x_train_new, y_train, x_test_new, y_test)
        print("\n Accuracy of the last epoch on the testing data (with all features): ", accuracy)
        print(f"The testing of the {k} most important weights took {time.time() - start_time} seconds")
        print(f"Accuracy of the last epoch on the testing data (with {k} features): ", accuracy_topk)

        # plot the features
        if args.plot_features:
            plot_features(data=args.data)

        if args.plot_importances:
            plot_importances(importances, k)