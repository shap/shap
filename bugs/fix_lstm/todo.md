- Bei RNN gibt es Abweichungen von 20% bei Assert Additivity. Also könnte ich mir die einzelnen Ops angucken, die hier vorkommen
und diese vergleichen mit GRU/LSTM, denn dort sind die Abweichungen geringer. Die Layers, die beide unterscheiden sollte ich genauer untersuchen.
- Auf master gibt es unterschiedliche Fehler für lstm/gru (no attribute between_tensors) und rnn (LookupError: gradient registry has no entry for: shap_TensorListStack).
- Wir könnten noch tf1 shap values mit tf2 shap values vergleichen. Hierzu müsste man ein Beispiel bekommen, das früher lauffähig war, es jetzt aber nicht mehr ist.

- [ ] nonlinear NN:
  - checke /home/tobias/programming/github/shap/.venv/lib/python3.11/site-packages/keras/src/utils/layer_utils.py(462)
  - checke /home/tobias/programming/github/shap/.venv/lib/python3.11/site-packages/keras/src/engine/functional.py(515)
  - (Pdb++) self._nodes_by_depth[0][0].is_input
    False
    (Pdb++) self._nodes_by_depth[5][0].is_input
    True
    (Pdb++) self._nodes_by_depth[4][0].is_input
    False
    (Pdb++) self._nodes_by_depth[3][0].is_input
    False
    (Pdb++) self._nodes_by_depth[2][0].is_input
    False
    (Pdb++) self._nodes_by_depth[1][0].is_input
    False
    (Pdb++) self._nodes_by_depth[4][1].is_input
    True
    (Pdb++) self._nodes_by_depth[4][1].layer
    <keras.src.engine.input_layer.InputLayer object at 0x7f1903905990>
    (Pdb++) self._nodes_by_depth[4][1].layer.name
    'input_2'
    (Pdb++) self._nodes_by_depth[5][0].layer.name
    'input_1'
  - debugge model(X) bzw. den Keras Code in dem das Netzwerk aufgebaut wird. So, dass ich verstehe wie dieses Graphennetzwerk gebaut wird und wie ich das halbwegs einfach durchlaufe.
  - lese https://www.tensorflow.org/guide/keras/functional_api