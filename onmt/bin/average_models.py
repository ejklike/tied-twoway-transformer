#!/usr/bin/env python
import argparse
import torch


def average_models(model_files, fp32=False):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        m = torch.load(model_file, map_location='cpu')
        model_weights = m['model']
        generator_x2y_weights = m['generator_x2y']
        generator_y2x_weights = m['generator_y2x']
        prior_weights = None
        if m.get('prior', None):
            prior_weights = m['prior']

        if fp32:
            for k, v in model_weights.items():
                model_weights[k] = v.float()
            for k, v in generator_x2y_weights.items():
                generator_x2y_weights[k] = v.float()
            for k, v in generator_y2x_weights.items():
                generator_y2x_weights[k] = v.float()
            if prior_weights is not None:
                for k, v in prior_weights.items():
                    prior_weights[k] = v.float()

        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator_x2y = generator_x2y_weights
            avg_generator_y2x = generator_y2x_weights
            avg_prior = prior_weights if prior_weights else None
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

            for (k, v) in avg_generator_x2y.items():
                avg_generator_x2y[k].mul_(i).add_(generator_x2y_weights[k]).div_(i + 1)
            
            for (k, v) in avg_generator_y2x.items():
                avg_generator_y2x[k].mul_(i).add_(generator_y2x_weights[k]).div_(i + 1)

            if prior_weights is not None:
                for (k, v) in avg_prior.items():
                    avg_prior[k].mul_(i).add_(prior_weights[k])\
                        .div_(i + 1)

    final = {"vocab": vocab, "opt": opt, "optim": None,
             "generator_x2y": avg_generator_x2y, 
             "generator_y2x": avg_generator_y2x, 
             "model": avg_model,
             "prior": avg_prior}
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models", "-m", nargs="+", required=True,
                        help="List of models")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    parser.add_argument("-fp32", "-f", action="store_true",
                        help="Cast params to float32")
    opt = parser.parse_args()

    final = average_models(opt.models, opt.fp32)
    torch.save(final, opt.output)


if __name__ == "__main__":
    main()
