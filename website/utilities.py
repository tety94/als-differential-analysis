def clean_numeric(value, cast_type=float):
    if value is None or value == '':
        return None
    return cast_type(value)