"""The real adapter transport must work from Hermes' threaded host lane."""
from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from tests.test_issue44_adapter import local_http, accepted
import noldomem


def test_threaded_host_write_succeeds_without_posix_timer():
    with local_http(lambda handler, calls: accepted('store')) as (url, calls, _):
        client = noldomem.NoldoMemHTTPClient(url, 'synthetic', .3)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(client.store, {'text': 'Synthetic host assertion'}).result(timeout=1)
        assert result['durable'] is True
        assert len(calls) == 1


def test_threaded_stalled_io_returns_within_shared_budget():
    def stall(handler, calls):
        time.sleep(.8)
        return accepted('store')
    with local_http(stall) as (url, calls, _):
        client = noldomem.NoldoMemHTTPClient(url, 'synthetic', .15)
        with ThreadPoolExecutor(max_workers=1) as executor:
            started = time.monotonic()
            with pytest.raises(RuntimeError, match='timed out'):
                executor.submit(client.store, {'text': 'Synthetic host assertion'}).result(timeout=1)
            assert time.monotonic() - started < .4
            assert len(calls) == 1
