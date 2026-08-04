"""Tests for shap/_serializable.py

Covers the Serializable base class, the Serializer context manager,
and the Deserializer context manager, including all encoder paths,
version validation, and round-trip serialization.

Works towards #3690.
"""

import io
import pickle

import cloudpickle
import numpy as np
import pytest

from shap._serializable import Deserializer, Serializable, Serializer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySerializable(Serializable):
    """A minimal Serializable subclass for testing save/load round-trips."""

    def __init__(self, value=None):
        self.value = value

    def save(self, out_file):
        super().save(out_file)
        with Serializer(out_file, "Dummy", 0) as s:
            s.save("value", self.value)

    @classmethod
    def load(cls, in_file, instantiate=True):
        obj = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "Dummy", 0, 0) as d:
            obj["value"] = d.load("value")
        if instantiate:
            return cls(**obj)
        return obj


# ===================================================================
# Serializable class
# ===================================================================


class TestSerializable:
    """Tests for the Serializable base class."""

    def test_save_writes_type(self):
        """save() pickles the class type as the first item."""
        buf = io.BytesIO()
        obj = Serializable()
        obj.save(buf)
        buf.seek(0)
        loaded_type = pickle.load(buf)
        assert loaded_type is Serializable

    def test_load_instantiate_false_returns_empty_dict(self):
        """load(instantiate=False) returns an empty dict for the base class."""
        buf = io.BytesIO()
        pickle.dump(Serializable, buf)
        buf.seek(0)
        result = Serializable.load(buf, instantiate=False)
        assert result == {}

    def test_instantiated_load_rejects_invalid_type(self):
        """_instantiated_load raises when the pickled type is not a Serializable subclass."""
        buf = io.BytesIO()
        pickle.dump(str, buf)  # str is not a Serializable subclass
        buf.seek(0)
        with pytest.raises(Exception, match="Invalid object type"):
            Serializable._instantiated_load(buf)

    def test_instantiated_load_none_returns_none(self):
        """_instantiated_load returns None when the stored type is None."""
        buf = io.BytesIO()
        pickle.dump(None, buf)
        buf.seek(0)
        result = Serializable._instantiated_load(buf)
        assert result is None


# ===================================================================
# Serializer
# ===================================================================


class TestSerializer:
    """Tests for the Serializer context manager."""

    def test_writes_block_header(self):
        """Entering the context should write serializer_version, block_name, block_version."""
        buf = io.BytesIO()
        with Serializer(buf, "TestBlock", 5):
            pass  # just enter and exit
        buf.seek(0)
        assert pickle.load(buf) == 0  # serializer_version
        assert pickle.load(buf) == "TestBlock"  # block_name
        assert pickle.load(buf) == 5  # block_version
        assert pickle.load(buf) == "END_BLOCK___"  # end token

    def test_save_encoder_none(self):
        """save() with encoder=None writes 'no_encoder' and no value."""
        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            s.save("my_key", "ignored_value", encoder=None)
        buf.seek(0)
        # skip block header
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        # data item
        assert pickle.load(buf) == "my_key"
        assert pickle.load(buf) == "no_encoder"
        # end block
        assert pickle.load(buf) == "END_BLOCK___"

    def test_save_encoder_false(self):
        """save() with encoder=False also writes 'no_encoder'."""
        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            s.save("k", 42, encoder=False)
        buf.seek(0)
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        assert pickle.load(buf) == "k"
        assert pickle.load(buf) == "no_encoder"

    def test_save_auto_pickle_for_primitives(self):
        """save() with encoder='auto' uses pickle.dump for int/float/str."""
        for value in [42, 3.14, "hello"]:
            buf = io.BytesIO()
            with Serializer(buf, "B", 0) as s:
                s.save("val", value)
            buf.seek(0)
            pickle.load(buf)  # serializer_version
            pickle.load(buf)  # block_name
            pickle.load(buf)  # block_version
            assert pickle.load(buf) == "val"
            assert pickle.load(buf) == "pickle.dump"
            assert pickle.load(buf) == value

    def test_save_auto_cloudpickle_for_complex(self):
        """save() with encoder='auto' uses cloudpickle for non-primitive objects."""
        value = [1, 2, 3]
        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            s.save("val", value)
        buf.seek(0)
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        assert pickle.load(buf) == "val"
        assert pickle.load(buf) == "cloudpickle.dump"
        loaded = cloudpickle.load(buf)
        assert loaded == value

    def test_save_serializable_instance(self):
        """save() with encoder='auto' for a Serializable instance uses .save."""
        obj = _DummySerializable(value=99)
        buf = io.BytesIO()
        with Serializer(buf, "Outer", 0) as s:
            s.save("child", obj)
        buf.seek(0)
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        assert pickle.load(buf) == "child"
        assert pickle.load(buf) == "serializable.save"

    def test_save_dot_save_encoder(self):
        """save() with encoder='.save' explicitly uses serializable.save."""
        obj = _DummySerializable(value=42)
        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            s.save("item", obj, encoder=".save")
        buf.seek(0)
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        assert pickle.load(buf) == "item"
        assert pickle.load(buf) == "serializable.save"

    def test_save_dot_save_fallback_for_non_serializable(self):
        """save() with encoder='.save' falls back to cloudpickle for non-Serializable objects."""
        value = {"key": "value"}
        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            s.save("item", value, encoder=".save")
        buf.seek(0)
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        assert pickle.load(buf) == "item"
        assert pickle.load(buf) == "cloudpickle.dump"
        loaded = cloudpickle.load(buf)
        assert loaded == value

    def test_save_custom_encoder(self):
        """save() with a callable encoder invokes the custom encoder function."""
        custom_values = []

        def my_encoder(value, stream):
            custom_values.append(value)
            pickle.dump(value * 2, stream)

        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            s.save("item", 10, encoder=my_encoder)
        assert custom_values == [10]
        buf.seek(0)
        pickle.load(buf)  # serializer_version
        pickle.load(buf)  # block_name
        pickle.load(buf)  # block_version
        assert pickle.load(buf) == "item"
        assert pickle.load(buf) == "custom_encoder"
        assert pickle.load(buf) == 20  # value * 2

    def test_save_unknown_encoder_raises(self):
        """save() with an unrecognized encoder string raises ValueError."""
        buf = io.BytesIO()
        with Serializer(buf, "B", 0) as s:
            with pytest.raises(ValueError, match="Unknown encoder type"):
                s.save("item", 42, encoder="unknown_encoder")


# ===================================================================
# Deserializer
# ===================================================================


class TestDeserializer:
    """Tests for the Deserializer context manager."""

    def _write_block_header(self, buf, *, serializer_version=0, block_name="B", block_version=0):
        """Helper to write a valid block header for testing deserialization."""
        pickle.dump(serializer_version, buf)
        pickle.dump(block_name, buf)
        pickle.dump(block_version, buf)

    def test_valid_header(self):
        """Deserializer enters successfully with a valid header."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)
        with Deserializer(buf, "B", 0, 0):
            pass  # should not raise

    def test_rejects_low_serializer_version(self):
        """Raises ValueError when serializer version is below minimum."""
        buf = io.BytesIO()
        self._write_block_header(buf, serializer_version=-1)
        buf.seek(0)
        with pytest.raises(ValueError, match="serializer version"):
            with Deserializer(buf, "B", 0, 0):
                pass

    def test_rejects_high_serializer_version(self):
        """Raises ValueError when serializer version is above maximum."""
        buf = io.BytesIO()
        self._write_block_header(buf, serializer_version=999)
        buf.seek(0)
        with pytest.raises(ValueError, match="serializer version"):
            with Deserializer(buf, "B", 0, 0):
                pass

    def test_rejects_wrong_block_name(self):
        """Raises ValueError when block name doesn't match."""
        buf = io.BytesIO()
        self._write_block_header(buf, block_name="WrongName")
        buf.seek(0)
        with pytest.raises(ValueError, match="supposed to be B"):
            with Deserializer(buf, "B", 0, 0):
                pass

    def test_rejects_low_block_version(self):
        """Raises ValueError when block version is below minimum."""
        buf = io.BytesIO()
        self._write_block_header(buf, block_version=0)
        buf.seek(0)
        with pytest.raises(ValueError, match="block version"):
            with Deserializer(buf, "B", min_version=1, max_version=5):
                pass

    def test_rejects_high_block_version(self):
        """Raises ValueError when block version is above maximum."""
        buf = io.BytesIO()
        self._write_block_header(buf, block_version=10)
        buf.seek(0)
        with pytest.raises(ValueError, match="block version"):
            with Deserializer(buf, "B", min_version=0, max_version=5):
                pass

    def test_load_pickle_dump(self):
        """load() can read a 'pickle.dump'-encoded data item."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        pickle.dump("my_item", buf)  # name
        pickle.dump("pickle.dump", buf)  # encoder
        pickle.dump(42, buf)  # value
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)
        with Deserializer(buf, "B", 0, 0) as d:
            result = d.load("my_item")
        assert result == 42

    def test_load_cloudpickle_dump(self):
        """load() can read a 'cloudpickle.dump'-encoded data item."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        pickle.dump("data", buf)  # name
        pickle.dump("cloudpickle.dump", buf)  # encoder
        cloudpickle.dump([1, 2, 3], buf)  # value
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)
        with Deserializer(buf, "B", 0, 0) as d:
            result = d.load("data")
        assert result == [1, 2, 3]

    def test_load_no_encoder(self):
        """load() returns None for 'no_encoder'-encoded data items."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        pickle.dump("key", buf)  # name
        pickle.dump("no_encoder", buf)  # encoder
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)
        with Deserializer(buf, "B", 0, 0) as d:
            result = d.load("key")
        assert result is None

    def test_load_wrong_name_raises(self):
        """load() raises ValueError when the stored name doesn't match."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        # Write a valid item under "actual_name"
        pickle.dump("actual_name", buf)  # name stored in stream
        pickle.dump("pickle.dump", buf)  # encoder
        pickle.dump(99, buf)  # value
        pickle.dump("END_BLOCK___", buf)  # end token
        buf.seek(0)
        # Asking to load "expected_name" but stream has "actual_name" — must raise ValueError
        with pytest.raises(ValueError):
            with Deserializer(buf, "B", 0, 0) as d:
                d.load("expected_name")

    def test_load_unsupported_encoder_raises(self):
        """load() raises ValueError for an unsupported encoder type."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        pickle.dump("item", buf)  # name
        pickle.dump("totally_unknown_encoder", buf)  # encoder
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)
        with pytest.raises(ValueError, match="Unsupported encoder"):
            with Deserializer(buf, "B", 0, 0) as d:
                d.load("item")

    def test_load_custom_decoder(self):
        """load() invokes a callable decoder when encoder is 'custom_encoder'."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        pickle.dump("item", buf)  # name
        pickle.dump("custom_encoder", buf)  # encoder tag
        pickle.dump(100, buf)  # data written by custom encoder
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)

        def my_decoder(stream):
            return pickle.load(stream) + 1

        with Deserializer(buf, "B", 0, 0) as d:
            result = d.load("item", decoder=my_decoder)
        assert result == 101

    def test_exit_skips_extra_items_before_end_block(self):
        """__exit__ skips unread data items until it finds END_BLOCK___."""
        buf = io.BytesIO()
        self._write_block_header(buf)
        # An unread data item
        pickle.dump("unread_item", buf)
        pickle.dump("pickle.dump", buf)
        pickle.dump("unread_value", buf)
        pickle.dump("END_BLOCK___", buf)
        buf.seek(0)
        # Entering and exiting without loading should not raise
        with Deserializer(buf, "B", 0, 0):
            pass


# ===================================================================
# Round-trip integration
# ===================================================================


class TestRoundTrip:
    """Full serialization round-trip tests."""

    def test_serializable_subclass_round_trip(self):
        """A Serializable subclass can save and load with correct data."""
        original = _DummySerializable(value=42)
        buf = io.BytesIO()
        original.save(buf)
        buf.seek(0)
        loaded = Serializable.load(buf)
        assert isinstance(loaded, _DummySerializable)
        assert loaded.value == 42

    def test_round_trip_none_value(self):
        """Round-trip with None value."""
        original = _DummySerializable(value=None)
        buf = io.BytesIO()
        original.save(buf)
        buf.seek(0)
        loaded = Serializable.load(buf)
        assert isinstance(loaded, _DummySerializable)
        assert loaded.value is None

    def test_round_trip_string_value(self):
        """Round-trip with string value."""
        original = _DummySerializable(value="test_string")
        buf = io.BytesIO()
        original.save(buf)
        buf.seek(0)
        loaded = Serializable.load(buf)
        assert loaded.value == "test_string"

    def test_round_trip_numpy_array(self):
        """Round-trip with a numpy array value (uses cloudpickle auto path)."""
        arr = np.array([1.0, 2.0, 3.0])
        original = _DummySerializable(value=arr)
        buf = io.BytesIO()
        original.save(buf)
        buf.seek(0)
        loaded = Serializable.load(buf)
        np.testing.assert_array_equal(loaded.value, arr)

    def test_serializer_deserializer_multi_items(self):
        """Serializer/Deserializer can handle multiple data items in a single block."""
        buf = io.BytesIO()
        with Serializer(buf, "Multi", 0) as s:
            s.save("int_val", 1)
            s.save("str_val", "hello")
            s.save("float_val", 3.14)

        buf.seek(0)
        with Deserializer(buf, "Multi", 0, 0) as d:
            assert d.load("int_val") == 1
            assert d.load("str_val") == "hello"
            assert d.load("float_val") == pytest.approx(3.14)
