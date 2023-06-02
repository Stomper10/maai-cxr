import timm

models = timm.list_models()
for m in models:
    print(m)
