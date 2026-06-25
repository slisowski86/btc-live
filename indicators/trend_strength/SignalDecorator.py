import functools

def signal(direction, signal_type="continuous", weight=1.0):
    """
    Decorator to mark a method as a trading signal generator.
    
    Parameters:
    direction : "long", "short", or "both"
    signal_type : "continuous" (always present when condition holds) or "discrete" (crossover/event)
    weight : relative importance when combining signals (default 1.0)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Return the original signal array
            return func(*args, **kwargs)
        # Attach metadata to the wrapper function
        wrapper._signal_meta = {
            'direction': direction,
            'type': signal_type,
            'weight': weight,
            'name': func.__name__
        }
        return wrapper
    return decorator