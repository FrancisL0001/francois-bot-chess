from statistics import mean

from chesstwin.eval.move_match import evaluate_model

# run from chess-twin root
if __name__=="__main__":
    print("="*30 + "\n" + " "*6 + "BEGIN TESTING" + " "*6 + "\n" + "="*30 + "\n") 
    history = evaluate_model()
    print("="*30 + "\n" + " "*5 + "TESTING COMPLETE" + " "*6 + "\n" + "="*30 + "\n") 
 
    print(f"Average: Top 1 Acc: {mean(history["top1_acc"])} - Top 3 Acc: {mean(history["top3_acc"])} - Top 5 Acc: {mean(history["top5_acc"])}\n") 
    print(f"Min: Top 1 Acc: {min(history["top1_acc"])} - Top 3 Acc: {min(history["top3_acc"])} - Top 5 Acc: {min(history["top5_acc"])}\n") 
    print(f"Max: Top 1 Acc: {max(history["top1_acc"])} - Top 3 Acc: {max(history["top3_acc"])} - Top 5 Acc: {max(history["top5_acc"])}\n") 

