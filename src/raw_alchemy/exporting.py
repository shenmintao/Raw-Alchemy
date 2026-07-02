def resolve_export_exposure(params, last_applied_ev):
    """Return (fast_path_exposure, full_path_exposure) for export paths."""
    manual_exposure = params.get('exposure', 0.0)
    if params.get('exposure_mode') == 'Auto':
        return last_applied_ev, None
    return manual_exposure, manual_exposure
