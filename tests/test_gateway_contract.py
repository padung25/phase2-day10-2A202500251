from reliability_lab.cache import ResponseCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.providers import FakeLLMProvider

VALID_ROUTE_PREFIXES = ("primary", "fallback", "static_fallback", "cache_hit")


def test_gateway_returns_response_with_route_reason() -> None:
    provider = FakeLLMProvider("primary", fail_rate=0.0, base_latency_ms=1, cost_per_1k_tokens=0.001)
    breaker = CircuitBreaker("primary", failure_threshold=2, reset_timeout_seconds=1)
    gateway = ReliabilityGateway([provider], {"primary": breaker}, ResponseCache(60, 0.5))
    result = gateway.complete("hello world")
    assert result.text
    route_prefix = result.route.split(":")[0]
    assert route_prefix in VALID_ROUTE_PREFIXES


def test_gateway_route_includes_provider_name() -> None:
    """Route reason must include the provider name (e.g. 'primary:myprovider')."""
    provider = FakeLLMProvider("gpt4", fail_rate=0.0, base_latency_ms=1, cost_per_1k_tokens=0.001)
    breaker = CircuitBreaker("gpt4", failure_threshold=2, reset_timeout_seconds=1)
    gateway = ReliabilityGateway([provider], {"gpt4": breaker})
    result = gateway.complete("test prompt")
    assert result.route == "primary:gpt4"


def test_gateway_fallback_route_includes_provider_name() -> None:
    """When primary is down, fallback route includes the backup provider name."""
    primary = FakeLLMProvider("primary", fail_rate=1.0, base_latency_ms=1, cost_per_1k_tokens=0.01)
    backup = FakeLLMProvider("backup", fail_rate=0.0, base_latency_ms=1, cost_per_1k_tokens=0.006)
    breaker_primary = CircuitBreaker("primary", failure_threshold=1, reset_timeout_seconds=60)
    breaker_backup = CircuitBreaker("backup", failure_threshold=5, reset_timeout_seconds=60)
    gateway = ReliabilityGateway(
        [primary, backup],
        {"primary": breaker_primary, "backup": breaker_backup},
    )
    # First request: primary fails, circuit opens, backup serves
    result = gateway.complete("hello")
    assert result.route.startswith("fallback:") or result.route == "static_fallback"


def test_gateway_static_fallback_when_all_providers_fail() -> None:
    """Static fallback is returned when every provider circuit is open."""
    provider = FakeLLMProvider("primary", fail_rate=1.0, base_latency_ms=1, cost_per_1k_tokens=0.001)
    breaker = CircuitBreaker("primary", failure_threshold=1, reset_timeout_seconds=60)
    gateway = ReliabilityGateway([provider], {"primary": breaker})
    # First call: primary fails, circuit opens; second call: circuit is open → static fallback
    gateway.complete("warm up")
    result = gateway.complete("should be static")
    assert result.route == "static_fallback"
    assert result.error is not None


def test_gateway_cache_hit_route() -> None:
    """A cache hit returns a route starting with 'cache_hit:'."""
    provider = FakeLLMProvider("primary", fail_rate=0.0, base_latency_ms=1, cost_per_1k_tokens=0.001)
    breaker = CircuitBreaker("primary", failure_threshold=2, reset_timeout_seconds=1)
    cache = ResponseCache(ttl_seconds=60, similarity_threshold=0.5)
    gateway = ReliabilityGateway([provider], {"primary": breaker}, cache)
    gateway.complete("hello world")        # populates cache
    result = gateway.complete("hello world")  # should hit cache
    assert result.cache_hit
    assert result.route.startswith("cache_hit:")


def test_circuit_opens_and_fallback_serves() -> None:
    """Primary fails N times → circuit opens → backup serves subsequent requests."""
    primary = FakeLLMProvider("primary", fail_rate=1.0, base_latency_ms=1, cost_per_1k_tokens=0.01)
    backup = FakeLLMProvider("backup", fail_rate=0.0, base_latency_ms=1, cost_per_1k_tokens=0.006)
    breaker_primary = CircuitBreaker("primary", failure_threshold=3, reset_timeout_seconds=60)
    breaker_backup = CircuitBreaker("backup", failure_threshold=5, reset_timeout_seconds=60)
    gateway = ReliabilityGateway(
        [primary, backup],
        {"primary": breaker_primary, "backup": breaker_backup},
    )
    # Drive primary to failure_threshold: each call tries primary, fails, then uses backup
    for _ in range(3):
        gateway.complete("probe")
    # Circuit should now be open for primary
    assert breaker_primary.state.value == "open", "primary circuit should be open after 3 failures"
    # Next request should skip primary (CircuitOpenError) and go directly to backup
    result = gateway.complete("after open")
    assert result.route.startswith("fallback:backup")
