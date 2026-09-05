from memory_demo import AppConfig, Database, MemoryApplication
from memory_demo.app import MemoryApplication as ApplicationImplementation
from memory_demo.config import AppConfig as ConfigImplementation
from memory_demo.contracts import RequestAnswerContract
from memory_demo.contracts.request import RequestAnswerContract as ContractImplementation
from memory_demo.database import Database as DatabaseImplementation


def test_runtime_public_api_exports_single_implementations() -> None:
    assert AppConfig is ConfigImplementation
    assert Database is DatabaseImplementation
    assert MemoryApplication is ApplicationImplementation


def test_contract_public_api_exports_single_implementation() -> None:
    assert RequestAnswerContract is ContractImplementation
