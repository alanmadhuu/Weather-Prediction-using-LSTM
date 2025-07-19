# Added detailed error logging
except Exception as e:
    logging.error(f"Prediction failed at step {step+1}: {str(e)}")
    break