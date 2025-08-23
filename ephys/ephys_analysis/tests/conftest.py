# conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--audit", action="store_true", default=False, help="run tests in audit mode")
    parser.addoption("--tensor_flow_test", action="store_true", default=False, help="run tensor flow tests")
    
