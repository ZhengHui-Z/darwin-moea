from darwin.ea.util.module_loader import loader


ndsortESS = getattr(loader("darwin.ea.operation.dominated.ndsortESS"), "ndsortESS")

ndsortDED = getattr(loader("darwin.ea.operation.dominated.ndsortDED"), "ndsortDED")

ndsortTNS = getattr(loader("darwin.ea.operation.dominated.ndsortTNS"), "ndsortTNS")
