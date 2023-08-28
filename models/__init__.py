from .AdaCLIP import AdaCLIP

def build_model(config, mode='adapt') -> AdaCLIP:
    if mode == 'adapt':
        return AdaCLIP(config, adapt=True)
    elif mode == 'finetune':
        return AdaCLIP(config, adapt=False)
    else:
        raise NotImplementedError()
