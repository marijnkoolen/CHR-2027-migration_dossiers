def map_to_preferred(label: str):
    """Map labels that are only used by a single annotator to a preferred label."""
    if label.startswith('Letter about procedure'):
        return 'Letter about procedure'
    elif label.startswith('Letter testimonial') or label.startswith('Testimonial') or label.lower().startswith('consent') or label.lower().startswith('divorce'):
        return 'Testimonial status (Application Documents)'
    elif label.lower().startswith('letter'):
        return 'Letter (Other)'
    elif 'proof' in label:
        return 'Testimonial status (Application Documents)'
    elif label == 'D2 (predecessor model 60)':
        return 'D.2'
    elif label.startswith('Form '):
        return label
    elif label == 'Trade Pro forma':
        return 'Form No. 136 (Trade Pro Forma)'
    elif label == 'Nominal Roll (excerpt)':
        return 'Form (Other)'
    elif label == 'Onbekend':
        return 'Form (Other)'
    elif label == 'check required of period in germany':
        return 'Letter about procedure (Security & Political Screening Documents)'
    elif label == 'Information on Professional Qualifications in the Netherlands':
        return 'Form No. 63 (Rijksarbeidsbureau - Qualifications)'
    elif label == 'Service Declaration':
        return 'Testimonial labour (Qualification & Employment Proof)'
    elif label == 'Report selection officer XS (Landing Permit)':
        return 'Report of selection and medical officers'
    elif 'certificate' in label.lower():
        return 'Testimonial status (Application Documents)'
    elif label == 'medical officers advise':
        return 'Report of selection and medical officers'
    elif label == 'telegram asking for health check':
        return 'Letter about procedure (Medical & Health Documents)'
    elif label == 'telegram notification':
        return 'Letter about procedure (Medical & Health Documents)'
    else:
        return 'Letter (Other)'

