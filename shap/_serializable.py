import pickle
#import types
import inspect
import logging
#import warnings
import numpy as np
import cloudpickle

log = logging.getLogger('shap')

class Serializable():
    """ This is the superclass of all serializable objects.
    """

    def save(self, out_file):
        """ Save the model to the given file stream.
        """
        pickle.dump(type(self), out_file)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ This is meant to be overriden by subclasses and called with super.

        We return constructor argument values when not being instantiated. Since there are no
        constructor arguments for the Serializable class we just return an empty dictionary.
        """
        if instantiate:
            return cls._instantiated_load(in_file)
        return {}

    @classmethod
    def _instantiated_load(cls, in_file, **kwargs):
        """ This is meant to be overriden by subclasses and called with super.

        We return constructor argument values (we have no values to load in this abstract class).
        """
        obj_type = pickle.load(in_file)
        if obj_type is None:
            return None

        if not inspect.isclass(obj_type) or (not issubclass(obj_type, cls) and (obj_type is not cls)):
            raise TypeError(f"Invalid object type loaded from file. {obj_type} is not a subclass of {cls}.")

        # here we call the constructor with all the arguments we have loaded
        constructor_args = obj_type.load(in_file, instantiate=False, **kwargs)
        used_args = inspect.getfullargspec(obj_type.__init__)[0]
        return obj_type(**{k: constructor_args[k] for k in constructor_args if k in used_args})


class Serializer():
    """ Save data items to an input stream.
    """
    def __init__(self, out_stream, block_name, version):
        self.out_stream = out_stream
        self.block_name = block_name
        self.block_version = version
        self.serializer_version = 0 # update this when the serializer changes

    def __enter__(self):
        log.debug("serializer_version = %d", self.serializer_version)
        pickle.dump(self.serializer_version, self.out_stream)
        log.debug("block_name = %s", self.block_name)
        pickle.dump(self.block_name, self.out_stream)
        log.debug("block_version = %d", self.block_version)
        pickle.dump(self.block_version, self.out_stream)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        log.debug("END_BLOCK___")
        pickle.dump("END_BLOCK___", self.out_stream)

    def save(self, name, value, encoder="auto"):
        """ Dump a data item to the current input stream.
        """
        log.debug("name = %s", name)
        pickle.dump(name, self.out_stream)
        if encoder is None or encoder is False:
            log.debug("encoder_name = %s", "no_encoder")
            pickle.dump("no_encoder", self.out_stream)
        elif callable(encoder):
            log.debug("encoder_name = %s", "custom_encoder")
            pickle.dump("custom_encoder", self.out_stream)
            encoder(value, self.out_stream)
        elif encoder == ".save" or (isinstance(value, Serializable) and encoder == "auto"):
            log.debug("encoder_name = %s", "serializable.save")
            pickle.dump("serializable.save", self.out_stream)
            if len(inspect.getfullargspec(value.save)[0]) == 3: # backward compat for MLflow, can remove 4/1/2021
                value.save(self.out_stream, value)
            else:
                value.save(self.out_stream)
        elif encoder == "auto":
            if isinstance(value, (int, float, str)):
                log.debug("encoder_name = %s", "pickle.dump")
                pickle.dump("pickle.dump", self.out_stream)
                pickle.dump(value, self.out_stream)
            else:
                log.debug("encoder_name = %s", "cloudpickle.dump")
                pickle.dump("cloudpickle.dump", self.out_stream)
                cloudpickle.dump(value, self.out_stream)
        else:
            raise ValueError(f"Unknown encoder type '{encoder}' given for serialization!")
        log.debug("value = %s", str(value))

class Deserializer():
    """ Load data items from an input stream.
    """

    def __init__(self, in_stream, block_name, min_version, max_version):
        self.in_stream = in_stream
        self.block_name = block_name
        self.block_min_version = min_version
        self.block_max_version = max_version

        # update these when the serializer changes
        self.serializer_min_version = 0
        self.serializer_max_version = 0

    def __enter__(self):

        # confirm the serializer version
        serializer_version = pickle.load(self.in_stream)
        log.debug("serializer_version = %d", serializer_version)
        if serializer_version < self.serializer_min_version:
            raise ValueError(
                f"The file being loaded was saved with a serializer version of {serializer_version}, " + \
                f"but the current deserializer in SHAP requires at least version {self.serializer_min_version}."
            )
        if serializer_version > self.serializer_max_version:
            raise ValueError(
                f"The file being loaded was saved with a serializer version of {serializer_version}, " + \
                f"but the current deserializer in SHAP only support up to version {self.serializer_max_version}."
            )

        # confirm the block name
        block_name = pickle.load(self.in_stream)
        log.debug("block_name = %s", block_name)
        if block_name != self.block_name:
            raise ValueError(
                f"The next data block in the file being loaded was supposed to be {self.block_name}, " + \
                f"but the next block found was {block_name}."
            )

        # confirm the block version
        block_version = pickle.load(self.in_stream)
        log.debug("block_version = %d", block_version)
        if block_version < self.block_min_version:
            raise ValueError(
                f"The file being loaded was saved with a block version of {block_version}, " + \
                f"but the current deserializer in SHAP requires at least version {self.block_min_version}."
            )
        if block_version > self.block_max_version:
            raise ValueError(
                f"The file being loaded was saved with a block version of {block_version}, " + \
                f"but the current deserializer in SHAP only support up to version {self.block_max_version}."
            )
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # confirm the block end token
        for _ in range(100):
            end_token = pickle.load(self.in_stream)
            log.debug("end_token = %s", end_token)
            if end_token == "END_BLOCK___":
                return
            else:
                self._load_data_value()
        raise ValueError(
            f"The data block end token wsa not found for the block {self.block_name}."
        )

    def load(self, name, decoder=None):
        """ Load a data item from the current input stream.
        """
        # confirm the block name
        loaded_name = pickle.load(self.in_stream)
        log.debug("loaded_name = %s", loaded_name)
        print("loaded_name", loaded_name)
        if loaded_name != name:
            raise ValueError(
                f"The next data item in the file being loaded was supposed to be {name}, " + \
                f"but the next block found was {loaded_name}."
            ) # We should eventually add support for skipping over unused data items in old formats...

        value = self._load_data_value(decoder)
        log.debug("value = %s", str(value))
        return value

    def _load_data_value(self, decoder=None):
        encoder_name = pickle.load(self.in_stream)
        log.debug("encoder_name = %s", encoder_name)
        if encoder_name == "custom_encoder" or callable(decoder):
            assert callable(decoder), "You must provide a callable custom decoder for the data item {name}!"
            return decoder(self.in_stream)
        if encoder_name == "no_encoder":
            return None
        if encoder_name == "serializable.save":
            return Serializable.load(self.in_stream)
        if encoder_name == "numpy.save":
            return np.load(self.in_stream)
        if encoder_name == "pickle.dump":
            return pickle.load(self.in_stream)
        if encoder_name == "cloudpickle.dump":
            return cloudpickle.load(self.in_stream)

        raise ValueError(f"Unsupported encoder type found: {encoder_name}")
