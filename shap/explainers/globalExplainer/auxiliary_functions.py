# Additional functions

class Data:
    def __init__(self, data, col_names):
        self.data = data
        #self.indx = data.index
        self.col_names = col_names
        self.n = data.shape[0]
        self.weights = np.ones(self.n)
        self.weights /= self.n

def convert_to_data(value):
    if isinstance(value, Data):
        return value
    elif type(value) == np.ndarray:
        return Data(value, [str(i) for i in range(value.shape[1])])
    elif str(type(value)).endswith("pandas.core.series.Series'>"):
        return Data(value.values.reshape((1,len(values))), value.index.tolist())
    elif str(type(value)).endswith("pandas.core.frame.DataFrame'>"):
        return Data(value.values, value.columns.tolist())
    else:
        assert False, str(type(value)) + "is currently not a supported format type"   

# Convert model to standard model class
class Model:
    def __init__(self, f, out_names):
        self.f = f
        self.out_names = out_names

def convert_to_model(value):
    if isinstance(value, Model):
        return value
    else:
        return Model(value, None)