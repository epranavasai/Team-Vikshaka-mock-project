from student_reply import generate_reply
from teacher_eval import evaluate_reply_with_teacher

def generate_final_reply(comment, embedder, index, df, threshold=0.7):
    student_reply = generate_reply(comment, embedder, index, df)
    score, feedback, final_reply = evaluate_reply_with_teacher(comment, student_reply, embedder, index, df, threshold)
    return final_reply
