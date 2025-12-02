import gsplat, gsplat.cuda._wrapper, inspect

print("gsplat.cuda._wrapper exports:")
for name in dir(gsplat.cuda._wrapper):
    if not name.startswith("_"):
        print(" -", name)

print("\nSignatures:")
for fn in ["world_to_cam", "quat_scale_to_covar_preci", "fully_fused_projection"]:
    try:
        print(fn, inspect.signature(getattr(gsplat, fn)))
    except:
        try:
            print(fn, inspect.signature(getattr(gsplat.cuda._wrapper, fn)))
        except:
            print(fn, "not found")
