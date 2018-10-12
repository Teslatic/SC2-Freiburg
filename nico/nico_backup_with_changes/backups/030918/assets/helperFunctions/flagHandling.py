def set_flag_every(self, var_input, ep):
    if var_input is None:
        return False
    else:
        return True if (ep % var_input == 0) else False

def set_flag_from(self, min_limit, ep):
    return False if (ep < min_limit) else True
