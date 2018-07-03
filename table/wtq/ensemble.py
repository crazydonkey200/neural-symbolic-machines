import codecs
import glob
import json
import os

import tensorflow as tf


# FLAGS
FLAGS = tf.app.flags.FLAGS  

tf.flags.DEFINE_string('eval_output_dir', '',
                       'Path to the folder with all the eval outputs.')

tf.flags.DEFINE_string('ensemble_output', '',
                       'Path to save the ensemble output')


def main(unused_argv):
    fns = [y for x in os.walk(FLAGS.eval_output_dir) for y in glob.glob(
        os.path.join(x[0], '*dev_programs_in_beam_0.json'))]

    model_predictions = []
    print('{} eval output found in {}'.format(
        len(fns), FLAGS.eval_output_dir))
    print('=' * 100)
    for fn in fns:
        print fn
        with open(fn, 'r') as f:
            model_predictions.append(json.load(f))
    print('=' * 100)
    ensemble_pred = {}
    for mp in model_predictions:
        for name, candidates in mp.iteritems():
            if name not in ensemble_pred:
                ensemble_pred[name] = {}
            for c in candidates:
                prob = c[-1]
                prog = ' '.join(c[0])
                answer = c[1]
                if prog in ensemble_pred[name]:
                    ensemble_pred[name][prog]['prob'] += prob
                else:
                    ensemble_pred[name][prog] = dict(prob=prob, answer=answer)


    with codecs.open(FLAGS.ensemble_output, 'w', encoding='utf-8') as f:
        for i in xrange(4344):
            name = u'nu-{}'.format(i)
            if name in ensemble_pred:
                answer = []
                max_prob = -1
                for c, info in ensemble_pred[name].iteritems():
                    if info['prob'] > max_prob:
                        answer = info['answer']
                        max_prob = info['prob']
                string = name
                for ans in answer:
                    if isinstance(ans, unicode):
                        string += u'\t{}'.format(ans)
                    else:
                        string += u'\t{}'.format(unicode(ans))
            else:
                string = name
            f.write(string + u'\n')
    

if __name__ == '__main__':  
    tf.app.run()
