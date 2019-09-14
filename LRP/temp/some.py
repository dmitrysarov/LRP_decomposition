from init_model import model
print(hasattr(model.model[0], 'foo'))
from over_load import model as model_over

print(hasattr(model_over.model[0], 'foo'))
