'''
Get filenames
'''
import tensorflow as tf


def get_modeldirname(opts):
    modeldir = opts['dirhead']

    # add if noise input is not there
    if opts ['z_off']:
        modeldir += "_noZ"
  
    # add normalization name
    if opts['applyinstancenorm']:
        modeldir += "_IN"
    elif opts['applybatchrenorm']:
        modeldir += "_BRN"
    elif opts['applybatchnorm']:
        modeldir += "_BN"
    elif opts['applygroupnorm']:
        modeldir += "_GN"
    elif opts['applyspectralnorm']:
        modeldir += "_SN"
    else:
        raise ValueError('Unknown option for normalization layer')

    # add labelsmooth
    if opts['D_real_target'] < 1.0 :
        modeldir += "_LabSmth" + str(opts['D_real_target'])

    # add gammatone init
    if opts['GT_init_G'] and opts['GT_init_D'] :
        modeldir += "_GDgt"
    elif opts['GT_init_D'] :
        modeldir += "_Dgt"
    elif opts['GT_init_G'] :
        modeldir += "_Ggt"
    

    # PreEmph Layer
    if opts['preemph_G'] and opts['preemph_D'] :
        modeldir += "_GD"
    elif opts['preemph_D'] :
        modeldir += "_D"
    elif opts['preemph_G'] :
        modeldir += "_G"
    if opts['preemph_G'] or opts['preemph_D'] :
        modeldir += "PreEm"

    return modeldir
    

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
  
