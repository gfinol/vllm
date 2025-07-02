from numpy import random


sleep_distribution = "uniform"  # Default sleep distribution
sleep_distribution_params = {
    "low": 0.1,
    "high": 0.5,
}  # Default parameters for uniform distribution
add_sleep_time = False # Default value for add_sleep_time
offload_to_ray = True # Default value for offload_to_ray

def set_add_sleep_time(add_sleep: bool) -> None:
    """
    Set the add sleep time flag for the model executor.

    Args:
        add_sleep (bool): Whether to add sleep time or not.
    """
    global add_sleep_time
    add_sleep_time = add_sleep

def set_offload_to_ray(offload: bool) -> None:
    """
    Set the offload to Ray flag for the model executor.

    Args:
        offload (bool): Whether to offload to Ray or not.
    """
    global offload_to_ray
    offload_to_ray = offload

def get_add_sleep_time() -> bool:
    """
    Get the add sleep time flag for the model executor.

    Returns:
        bool: Whether to add sleep time or not.
    """
    return add_sleep_time

def get_offload_to_ray() -> bool:
    """
    Get the offload to Ray flag for the model executor.

    Returns:
        bool: Whether to offload to Ray or not.
    """
    return offload_to_ray

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)


def set_sleep_distribution(
    in_sleep_distribution: str,
    in_sleep_distribution_params: dict,
) -> None:
    """
    Set the sleep distribution for the model executor.
    Args:
        in_sleep_distribution (str): The sleep distribution to set.
        in_sleep_distribution_params (dict): The parameters for the sleep distribution.
    """

    global sleep_distribution
    global sleep_distribution_params

    if in_sleep_distribution not in ["uniform", "normal"]:
        raise ValueError(f"Unsupported sleep distribution: {in_sleep_distribution}")

    sleep_distribution = in_sleep_distribution
    sleep_distribution_params = in_sleep_distribution_params


def get_sleep_time_function():
    """
    Get the sleep time function based on the sleep distribution.

    Returns:
        function: The sleep time function.
    """
    # Placeholder for actual implementation

    if sleep_distribution == "uniform":
        def sleep_time_uniform():
            random.uniform(
                sleep_distribution_params["low"],
                sleep_distribution_params["high"],
            )
        return sleep_time_uniform()

    elif sleep_distribution == "normal":
        def sleep_time_normal():
            random.normal(
                loc=sleep_distribution_params["loc"],
                scale=sleep_distribution_params["scale"],
            )
        return sleep_time_normal
    else:
        raise ValueError(f"Unsupported sleep distribution: {sleep_distribution}")