from autoemulate.experimental.compare import AutoEmulate


def test_compare(sample_data_y2d):
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)
    results = ae.compare(10)
    print(results)
