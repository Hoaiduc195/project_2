import importlib
import pkgutil


MODEL_REGISTRY = {}


def register_model(name):
	def decorator(model_cls):
		MODEL_REGISTRY[name] = model_cls
		return model_cls
	return decorator


def discover_model_plugins(package_name='part2.models'):
	package = importlib.import_module(package_name)
	for _, module_name, _ in pkgutil.iter_modules(package.__path__):
		importlib.import_module(f"{package_name}.{module_name}")


def list_registered_models():
	return sorted(MODEL_REGISTRY.keys())


def build_models(model_names=None, model_params=None):
	discover_model_plugins()

	if model_params is None:
		model_params = {}

	if model_names is None:
		model_names = list_registered_models()

	models = []
	for model_name in model_names:
		if model_name not in MODEL_REGISTRY:
			raise ValueError(
				f"Unknown model '{model_name}'. "
				f"Available models: {list_registered_models()}"
			)
		model_cls = MODEL_REGISTRY[model_name]
		params = model_params.get(model_name, {})
		models.append(model_cls(**params))

	return models
