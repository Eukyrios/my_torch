import pytest

from my_torch.optimizers.schedule import Constant, ExponentialDecay, Schedule, StepDecay, TimeDecay


def test_constant_keeps_the_initial_rate():
    assert [Constant()(0.1, epoch) for epoch in (0, 5, 100)] == [0.1, 0.1, 0.1]


def test_time_decay_divides_by_one_plus_decay_times_epoch():
    schedule = TimeDecay(decay=0.5)
    assert [schedule(0.1, epoch) for epoch in (0, 1, 2)] == pytest.approx([0.1, 0.1 / 1.5, 0.1 / 2])


def test_exponential_decay_multiplies_by_the_factor_each_epoch():
    schedule = ExponentialDecay(factor=0.9)
    assert [schedule(0.1, epoch) for epoch in (0, 1, 2)] == pytest.approx([0.1, 0.09, 0.081])


def test_step_decay_drops_every_few_epochs():
    schedule = StepDecay(factor=0.5, every=10)
    assert [schedule(0.1, epoch) for epoch in (0, 9, 10, 19, 20)] == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025])


@pytest.mark.parametrize("schedule", [TimeDecay(0.1), ExponentialDecay(0.9), StepDecay(0.5, 3)])
def test_every_decay_starts_at_the_initial_rate_and_never_goes_up(schedule):
    rates = [schedule(0.1, epoch) for epoch in range(20)]
    assert rates[0] == pytest.approx(0.1)
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))


@pytest.mark.parametrize("schedule, name", [(Constant(), "constant"), (TimeDecay(0.1), "time_decay"), (ExponentialDecay(0.9), "exponential_decay"), (StepDecay(0.5, 3), "step_decay")])
def test_each_schedule_has_its_name(schedule, name):
    assert schedule.name == name


def test_time_decay_rejects_a_negative_decay():
    with pytest.raises(ValueError, match="must not be negative"):
        TimeDecay(decay=-0.1)


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.5])
def test_rejects_a_factor_outside_0_and_1(factor):
    with pytest.raises(ValueError, match=r"decay factor must be in \(0, 1\]"):
        ExponentialDecay(factor)
    with pytest.raises(ValueError, match=r"decay factor must be in \(0, 1\]"):
        StepDecay(factor, every=2)


def test_step_decay_rejects_a_step_below_one_epoch():
    with pytest.raises(ValueError, match="at least 1 epoch"):
        StepDecay(0.5, every=0)


def test_the_base_class_cannot_be_used_on_its_own():
    with pytest.raises(TypeError):
        Schedule("schedule")


def test_the_base_rate_is_abstract():
    with pytest.raises(NotImplementedError):
        Schedule.rate(Constant(), 0.1, 0)
