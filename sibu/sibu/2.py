#rnn
model_rnn = BuildNetwork1(input_shape=input_shape, class_num=2)
model_rnn.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rnn.summary()
model_rnn.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rnn = model_rnn.evaluate(x_test, y_test, batch_size=64)

model_rnn.save_weights("model_rnn(12).h5")

##############################################################################################################################################
#lstm
model_lstm = BuildNetwork2(input_shape=input_shape, class_num=2)
model_lstm.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lstm.summary()
model_lstm.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lstm = model_lstm.evaluate(x_test, y_test, batch_size=64)

model_lstm.save_weights("model_lstm(12).h5")

##############################################################################################################################################
#gru
model_gru = BuildNetwork3(input_shape=input_shape, class_num=2)
model_gru.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gru.summary()
model_gru.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gru = model_gru.evaluate(x_test, y_test, batch_size=64)

model_gru.save_weights("model_gru(12).h5")

##############################################################################################################################################

#rnn + cooc
model_rc = BuildNetwork4(input_shape=input_shape, class_num=2)
model_rc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc.summary()
model_rc.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc = model_rc.evaluate(x_test, y_test, batch_size=64)

model_rc.save_weights("model_rc(12).h5")

##############################################################################################################################################
#lstm + cooc
model_lc = BuildNetwork5(input_shape=input_shape, class_num=2)
model_lc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc.summary()
model_lc.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc = model_lc.evaluate(x_test, y_test, batch_size=64)

model_lc.save_weights("model_lc(12).h5")

##############################################################################################################################################

#gru + cooc
model_gc = BuildNetwork6(input_shape=input_shape, class_num=2)
model_gc.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc.summary()
model_gc.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc = model_gc.evaluate(x_test, y_test, batch_size=64)

model_gc.save_weights("model_gc(12).h5")

##############################################################################################################################################
#rnn + Conv
model_rc2 = BuildNetwork7(input_shape=input_shape, class_num=2)
model_rc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_rc2.summary()
model_rc2.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_rc2 = model_rc2.evaluate(x_test, y_test, batch_size=64)

model_rc2.save_weights("model_rc2(12).h5")

##############################################################################################################################################
#lstm + Conv
model_lc2 = BuildNetwork8(input_shape=input_shape, class_num=2)
model_lc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_lc2.summary()
model_lc2.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_lc2 = model_lc2.evaluate(x_test, y_test, batch_size=64)

model_lc2.save_weights("model_lc2(12).h5")

##############################################################################################################################################
#gru + Conv
model_gc2 = BuildNetwork9(input_shape=input_shape, class_num=2)
model_gc2.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_gc2.summary()
model_gc2.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_gc2 = model_gc2.evaluate(x_test, y_test, batch_size=64)

model_gc2.save_weights("model_gc2(12).h5")

#############################################################################################################################################
#hlac + deep cnn
model_hd = BuildNetwork10(input_shape=input_shape, class_num=2)
model_hd.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hd.summary()
model_hd.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hd = model_hd.evaluate(x_test, y_test, batch_size=64)

model_hd.save_weights("model_hd(12).h5")

##############################################################################################################################################
#cooc +lenet
model_cl = BuildNetwork11(input_shape=input_shape, class_num=2)
model_cl.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_cl.summary()
model_cl.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_cl = model_cl.evaluate(x_test, y_test, batch_size=64)

model_cl.save_weights("model_c1(12).h5")

##############################################################################################################################################
#hlac + gru
model_hg = BuildNetwork12(input_shape=input_shape, class_num=2)
model_hg.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=[metrics.categorical_accuracy],sample_weight_mode="None")
#model_hg.summary()
model_hg.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0, callbacks=[es_cb], validation_data=(x_test, y_test),validation_split=0)
score_hg = model_hg.evaluate(x_test, y_test, batch_size=64)

model_hg.save_weights("model_hg(12).h5")

##############################################################################################################################################

print("--------------64--------------")
print(model_rnn.metrics_names)
print(score_rnn)
print(model_lstm.metrics_names)
print(score_lstm)
print(model_gru.metrics_names)
print(score_gru)
print(model_rc.metrics_names)
print(score_rc)
print(model_lc.metrics_names)
print(score_lc)
print(model_gc.metrics_names)
print(score_gc)
print(model_rc2.metrics_names)
print(score_rc2)
print(model_lc2.metrics_names)
print(score_lc2)
print(model_gc2.metrics_names)
print(score_gc2)
print(model_hd.metrics_names)
print(score_hd)
print(model_cl.metrics_names)
print(score_cl)
print(model_hg.metrics_names)
print(score_hg)
