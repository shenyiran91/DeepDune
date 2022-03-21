import argparse
from datetime import datetime
import json
from data import get_train_valid
from loss import losses
from model import models
from tensorflow import keras
from tensorflow import config
from utils.metrics import recall, precision, f1
from utils.optimizers import get_optimizer
from utils.util import set_seed


def main(args):
    # setup GPU card
    gpus = config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(gpu)
            # dynamic memory allocation
            config.experimental.set_memory_growth(gpu, True)

    # get data
    data_path = '../small_data/mars_dunes.npz'
    data_train, data_valid, labels_train, labels_valid = get_train_valid(
        data_path)

    # build model
    model = getattr(models, args.model)()
    model.compile(loss=getattr(losses, args.loss),
                  optimizer=get_optimizer(args.opt, args.lr, args.wd),
                  metrics=['accuracy', recall, precision, f1, keras.metrics.MeanIoU(num_classes=2)])

    # train and evaluate
    fit = model.fit(data_train, labels_train,
                    validation_data=(data_valid, labels_valid),
                    epochs=args.epochs,
                    batch_size=args.bs)

    # save record
    timestamp = timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    exp_name = args.model + '_' + args.loss + '_' + args.opt + '_lr' + str(args.lr) + '_wd' + str(args.wd) + \
        '_bs' + str(args.bs) + '_ep' + str(args.epochs) + '_seed' + str(args.seed) + '_' + timestamp
    record_path = f"../records/{exp_name}.json"
    with open(record_path, "w") as file:
        json.dump(fit.history, file)
    print(f"Record saved at {record_path}.")

    # save model
    if args.save_model:
        model_path = f"../saved_models/{exp_name}.h5"
        keras.models.save_model(model, model_path, save_format='h5')
        print(f"Model saved at {model_path}.")

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')
    parser.add_argument('--loss', default='cross_entropy',
                        type=str, help='loss')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--model', default='Unet',
                        type=str, help="segmentation model")
    parser.add_argument('--opt', default='adam',
                        type=str, help='optimizer type')
    parser.add_argument('--seed', default=888, type=int, help='random seed')
    parser.add_argument('--save_model', help='save model', action='store_true')
    parser.add_argument('--wd', default=0, type=float, help="weight decay parameter")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
