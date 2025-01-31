import json


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/mdetr_processed_json_annotations', help='annotation_file')
    
    args = parser.parse_args()
    
    gts = [json.loads(q) for q in open(args.annotation_file, "r")]
    answers = [json.loads(q) for q in open(args.prediction_file, "r")]

    num_correct, num_total = 0, 0
    index, round_correct = 0, 0
    for gt, ans in zip(gts, answers):
        index += 1
        if ans['text'].lower().strip().strip('\n').strip('.') in gt['answers']:
            round_correct += 1
        if index == 2:
            index = 0
            if round_correct == 2:
                num_correct += 1
            round_correct = 0

            num_total += 1
    print(f"The accuracy is {num_correct/num_total}")






