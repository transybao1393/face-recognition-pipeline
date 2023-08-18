from ..pipeline import Pipeline

class Predict():
    def __init__(self) -> None:
        pass

    def exec():
        # pipeline start here
        step1 = ""
        step2 = ""
        
        pipeline = (step1 | step2)
        # Iterate through pipeline
        try:
            # Iterate through pipeline
            for _ in pipeline:
                pass
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        finally:
            print(f"[INFO] End of pipeline")

# Image preprocessing including image resolution improve, denoise, dehazing
class PredictCamera(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)

# Image preprocessing including image resolution improve, denoise, dehazing
class PredictVideo(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)

# Image preprocessing including image resolution improve, denoise, dehazing
class PredictDeviceWebcam(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)
    
