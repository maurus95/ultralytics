from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="yolov8l.yaml")
    parser.add_argument("-d", "--data", default="yolov8_eb_data.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max_epoch", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    model = YOLO(args.model)

    if args.test:
        model.val(data=args.data, split="test", amp=False)
    elif args.resume:
        model.train(
            data=args.data,
            epochs=args.max_epoch,
            batch=args.batch_size,
            device=[int(d) for d in args.device.split(",")],
            amp=args.amp,
            resume=True,
        )
    else:
        model.train(
            data=args.data,
            epochs=args.max_epoch,
            batch=args.batch_size,
            device=[int(d) for d in args.device.split(",")],
            amp=args.amp,
        )
        model.val(data=args.data, split="val")
