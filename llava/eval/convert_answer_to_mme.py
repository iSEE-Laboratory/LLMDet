import json

def write_txt(file_name, lines):
    with open(file_name, 'w') as f:
        for line in lines:
            f.write(line)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', help='annotation_file')
    args = parser.parse_args()
    
    questions = [json.loads(q) for q in open(args.annotation_file, "r")]
    answer = [json.loads(q) for q in open(args.prediction_file, "r")]
    assert len(questions) == len(answer)



    lines = []
    current_file = None
    for ques, ans in zip(questions, answer):
        file = ques['image'].split('/')[0]
        if current_file is None:
            current_file = file
        elif current_file != file:
            write_txt(current_file + '.txt', lines)
            lines = []
            current_file = file
        ans_text = ans['text'].strip().strip('\n')
        lines.append(ques["question_id"] + '\t' + ques['text'] + '\t' + ques['answer'] + '\t' + ans_text + '\n')

    if len(lines):
        write_txt(current_file + '.txt', lines)



