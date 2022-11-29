"""Utilities for logging messages and data."""

import os.path
from collections import defaultdict
import numpy as np
import torch


class Logger:
    """Simple logger that saves to a file and stdout."""

    def __init__(self, out_file, is_primary, print_stdout=True):
        """Save logging info to out_file."""
        self.is_primary = is_primary
        self.print_stdout = print_stdout
        if is_primary:
            if os.path.exists(out_file):
                raise ValueError(f'Log file {out_file} already exists')
            self.log_file = open(out_file, 'w')
        else:
            self.log_file = None

    def log(self, message):
        """Log message."""
        if self.log_file is not None:
            # Only the primary writes the log.
            self.log_file.write(message + '\n')
            self.log_file.flush()
            if self.print_stdout:
                print(message, flush=True)

    def close(self):
        """Close the log."""
        if self.log_file is not None:
            self.log_file.close()


class DataLogger:
    """Class for saving quantities from networks."""

    def __init__(self, net, save_weights, save_grads):
        """Save quantities from network net.

        If save_weights is True, every time record is called, all
        parameters in net will be saved.

        If save_grads is True, every time record is called, all
        parameter gradients in net will be saved.

        Weights and gradients are only saved when the net is in
        training mode.

        """
        self.net = net
        self.save_weights = save_weights
        self.save_grads = save_grads
        self.saved_tensors = defaultdict(list)  # name -> list of tensors.
        self.attribs_to_save = []
        self.has_batch_dim = {}  # Whether an attribute has a batch dimension.
        self.val_only = {}  # Whether to only record attributes in validation.

    def record_module_attribute(self, attrib, has_batch_dim=True,
                                val_only=True):
        """Record attrib from all modules in the network that have it.

        attrib should be the name of a member of modules. Every time
        record is called, this will copy and save the value of that
        attribute from every module in the network.

        has_batch_dim is whether the attribute has a batch dimension,
        in which case tensors will be concatenated along the first
        dimension when saving.

        val_only is whether the attribute is always recorded, or only
        recorded during validation.

        """
        if attrib in self.attribs_to_save:
            raise RuntimeError(f'Trying to save "{attrib}" which is already'
                               ' being saved')
        self.attribs_to_save.append(attrib)
        self.has_batch_dim[attrib] = has_batch_dim
        self.val_only[attrib] = val_only

    def record_tensor(self, name, tensor, has_batch_dim=True):
        """Record a specific tensor using name."""
        self.saved_tensors[name].append(tensor.detach().cpu().clone())
        self.has_batch_dim[name] = has_batch_dim

    def clear(self):
        """Drop all recorded data."""
        self.saved_tensors.clear()

    def record_params(self):
        """Record layer weights and gradients, if requested."""
        if not self.save_weights and not self.save_grads:
            return
        if not self.net.training:
            return
        for name, param in self.net.named_parameters():
            if self.save_weights:
                self.saved_tensors[name + '.weight'].append(param.detach().cpu().clone())
                self.has_batch_dim[name + '.weight'] = False
            if self.save_grads and param.grad is not None:
                self.saved_tensors[name + '.grad'].append(param.grad.detach().cpu().clone())
                self.has_batch_dim[name + '.grad'] = False

    def record(self):
        """Record all requested quantities from the network."""
        self.record_params()
        for name, module in self.net.named_modules():
            for attrib in self.attribs_to_save:
                if attrib in self.val_only and self.val_only[attrib] and self.net.training:
                    continue
                if hasattr(module, attrib):
                    val = getattr(module, attrib).detach().cpu().clone()
                    self.saved_tensors[name + '.' + attrib].append(val)
                    self.has_batch_dim[name + '.' + attrib] = self.has_batch_dim[attrib]

    def save(self, job_id, out_dir, epoch):
        """Save all recorded data.

        Data will either be concatenated along a new dimension.

        Data will be saved 'out_dir/job_id_name_epoch' in numpy format.

        After saving, all data will be cleared.

        """
        for name, tensor_list in self.saved_tensors.items():
            if self.has_batch_dim[name]:
                data = torch.cat(tensor_list)
            else:
                data = torch.stack(tensor_list)
            data = data.numpy()
            np.save(os.path.join(out_dir, f'{job_id}_{name}_{epoch}'), data)
        self.clear()
