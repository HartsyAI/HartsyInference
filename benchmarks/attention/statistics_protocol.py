"""Predeclared paired log-ratio analysis; confirmation is never pooled into the interval."""
import math
import random


def percentile(values, quantile):
    ordered = sorted(values)
    at = (len(ordered) - 1) * quantile
    low, high = math.floor(at), math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def analyze(sessions, confirmation, claim, seed=274019, resamples=20000):
    if len(sessions) != 3 or any(len(session) != 5 for session in sessions) or len(confirmation) != 5:
        raise ValueError('Exactly three five-pair sessions and one five-pair confirmation are required')
    if claim not in ('improvement', 'noninferiority', 'regression'):
        raise ValueError('Unknown claim')
    if resamples != 20000 or seed != 274019:
        raise ValueError('Resampling settings differ from the frozen protocol')
    def logs(pairs):
        result = []
        for pair in pairs:
            if pair.get('valid') is not True:
                raise ValueError('Quality, route, memory, or cohort validation failed')
            candidate, reference = pair['candidate_ms'], pair['reference_ms']
            if not all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in (candidate, reference)):
                raise ValueError('Latency must be positive and finite')
            result.append(math.log(candidate / reference))
        return result
    groups, holdout = [logs(session) for session in sessions], logs(confirmation)
    rng = random.Random(seed)
    bootstrap = [math.exp(sum(sum(rng.choices(group, k=5)) / 5 for group in groups) / 3)
                 for _ in range(resamples)]
    ratio = math.exp(sum(sum(group) / 5 for group in groups) / 3)
    confirmation_ratio = math.exp(sum(holdout) / 5)
    lower, upper = percentile(bootstrap, .025), percentile(bootstrap, .975)
    upper_one, lower_one = percentile(bootstrap, .95), percentile(bootstrap, .05)
    passed = {'improvement': upper < 1 and confirmation_ratio < 1,
              'noninferiority': upper_one <= 1.03 and confirmation_ratio <= 1.03,
              'regression': lower_one > 1.03 and confirmation_ratio > 1.03}[claim]
    return {'ratio': ratio, 'ci95_two_sided': [lower, upper], 'upper95_one_sided': upper_one,
            'lower95_one_sided': lower_one, 'confirmation_ratio': confirmation_ratio,
            'claim': claim, 'passed': passed, 'inferential_pairs': 15, 'confirmation_pairs': 5}
