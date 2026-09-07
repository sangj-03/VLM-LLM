#!/usr/bin/env python3
from __future__ import annotations

def _v(row,key,default='uncertain'):
    try: value=row[key]
    except Exception: value=default
    if value is None: return default
    return str(value).strip().lower()

def safety_supervisor(row, threshold):
    probability=float(row.get('tcn_peak_probability',row.get('tcn_probability',0.0)))
    purposeful=_v(row,'purposeful_action')
    self_response=_v(row,'self_response')
    external=_v(row,'external_contact')
    contact_response=_v(row,'response_after_contact','not_applicable')
    visibility=_v(row,'visibility','poor')
    support_loss=_v(row,'loss_of_support')
    recovery=_v(row,'deliberate_recovery','not_applicable')
    supported_end=_v(row,'supported_at_end')
    normalized_recovery='not_applicable' if support_loss!='yes' else recovery
    if probability < float(threshold):
        return 'active','tcn_clear',normalized_recovery
    if visibility != 'good':
        return 'reduced','insufficient_visibility',normalized_recovery
    if support_loss=='yes' and supported_end=='no':
        return 'no_visible_response','unrecovered_support_loss',normalized_recovery
    if purposeful=='no' and self_response=='no':
        return 'no_visible_response','no_self_initiated_response',normalized_recovery
    if external=='yes' and contact_response=='no':
        return 'no_visible_response','no_response_to_external_contact',normalized_recovery
    if purposeful=='yes' and self_response=='yes' and support_loss=='no' and supported_end=='yes':
        return 'active','qwen_active_reject',normalized_recovery
    return 'reduced','mixed_reduced_evidence',normalized_recovery
