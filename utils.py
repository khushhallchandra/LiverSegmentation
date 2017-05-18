import json


def save_model(model, history, name):
    h = json.dumps(history.history)
    with open(name + '_history.txt', 'w') as f:
        f.write(h)
    model.save_weights(name + '_weights.h5')
    arch_str = model.to_json()
    with open(name + '_arch.txt', 'w') as f:
        f.write(arch_str)


def load_model(name):
    # TODO
    pass
