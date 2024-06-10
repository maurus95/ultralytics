from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="yolov8l.yaml")
    parser.add_argument("-d", "--data", default="etram_histo.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--max_epoch", default=20, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--optimizer", default="auto")

    args = parser.parse_args()
    model = YOLO(args.model)

    if args.test:
        model.val(data=args.data, 
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            amp=args.amp,
            cache=args.cache,
            split="test")
    elif args.resume:
        model.train(resume=True)
    else:
        model.train(
            data=args.data,
            epochs=args.max_epoch,
            batch=args.batch_size,
            device=[int(d) for d in args.device.split(",")],
            workers=args.workers,
            resume=args.resume,
            amp=args.amp,
            cache=args.cache,
            optimizer=args.optimizer,
        )