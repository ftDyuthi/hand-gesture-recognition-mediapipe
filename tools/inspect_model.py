import torch
import pprint
p = r"c:\Users\pufre\Downloads\CodingProjects\hand-gesture-recognition-mediapipe\model\keypoint_sequence_classifier\keypoint_sequence_classifier.pth"
print('Loading:', p)
try:
    o = torch.load(p, map_location='cpu')
    print('Loaded type:', type(o))
    if isinstance(o, dict):
        print('==> torch.load returned dict (likely state_dict)')
        print('Keys count:', len(o))
        keys = list(o.keys())
        pprint.pprint(keys[:100])
    elif isinstance(o, torch.nn.Module):
        print('==> torch.load returned nn.Module instance')
        print('Class:', o.__class__)
        # try to access scripted attributes if present
        for attr in ('sequence_length','feature_dim'):
            if hasattr(o, attr):
                print(f'Has attribute {attr}:', getattr(o, attr))
        # show some attributes
        attrs = [a for a in dir(o) if not a.startswith('_')]
        print('Sample attrs:', attrs[:50])
    else:
        # print repr summary
        print('torch.load returned object repr:')
        print(repr(o)[:1000])
except Exception as e:
    print('ERROR during torch.load:', e)
    import traceback; traceback.print_exc()
