"""
Tests for `cpm` module.
"""

import pytest
import cpm


class TestCompact(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_metadata(self):
        assert hasattr(
            cpm, "__version__"
        ), "Module does not have a __version__ attribute"
        assert hasattr(
            cpm, "__author__"
        ), "Module does not have an __author__ attribute"
        assert hasattr(
            cpm, "__license__"
        ), "Module does not have a __license__ attribute"

    def test_version_format(self):
        version = cpm.__version__
        assert isinstance(version, str), "Version is not a string"
        assert version.count(".") == 2, "Version format is incorrect"

    def test_author_format(self):
        author = cpm.__author__
        assert isinstance(author, str), "Author is not a string"
        assert len(author) > 0, "Author string is empty"

    def test_license_format(self):
        license = cpm.__license__
        assert isinstance(license, str), "License is not a string"
        assert len(license) > 0, "License string is empty"

    @classmethod
    def teardown_class(cls):
        pass


if __name__ == "__main__":
    pytest.main()
