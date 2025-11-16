class VAccessError(Exception):
    """Base exception for V-Access"""
    pass

class ClinicNotFoundError(VAccessError):
    """No clinics found for location"""
    pass

class AppointmentBookingError(VAccessError):
    """Failed to book appointment"""
    pass

class ValidationError(VAccessError):
    """Data validation failed"""
    pass

# Use specific exceptions instead of generic Exception