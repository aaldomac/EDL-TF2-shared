# Usage of the code #
All modules are presented in the EDL_modules folder. The main modules are:
- EDL_callbacks.py: contains the callbacks for updating the epoch in the loss while training.
- EDL_models.py: contains the models for the Evidential Networks. The model has a linear output layer.
- EDL_losses.py: contains the losses for the Evidential Networks. The output of the model is mapped to evidence through a ReLU and then to a Dirichlet inside the loss.
- EDL_metrics.py: contains the metrics for the Evidential Networks these are the accuracy, the mean evidence, the mean evidence of success and the mean evidence of fails. The output of the model is modified inside each metric depending on the metric's needs.
- EDL_utils.py: contains the auxiliary functions for importing datasets and plotting results.

All of these models are subclasses of their respective classes in the Keras module in order to have a better 
flexibility. 

To run the EDL framework with mse loss it is only necessary to turn the TEST_EQ5=True in the EDL_script.py file
and run it. With this, one runs the following chunk of code:
```python
    if TEST_EQ5:
        # Create the LeNet_EDL network, compile and train
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        metrics = [EDL_Accuracy(), EDL_mean_ev(logits2evidence), EDL_mean_ev_succ(logits2evidence),
                   EDL_mean_ev_fail(logits2evidence)]
        LeNet_EDL_model = LeNet_EDL()
        LeNet_EDL_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)
        history = LeNet_EDL_model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test),
                                      batch_size=batch_size, callbacks=[UpdateEpochCallback(loss_fn)])

        # Plot the training history
        draw_EDL_results(history)

        # Plot results on rotating image
        rotating_image_classification(digit_one, LeNet_EDL_model, uncertainty=True)
```