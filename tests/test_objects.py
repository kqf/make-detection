from detasks.objects import distribution_count, distribution_size, make_objects


def test_objects():
    objects = make_objects(
        n_samples=100,
        distribution_count=distribution_count,
        distribution_size=distribution_size,
    )
    print(objects)
