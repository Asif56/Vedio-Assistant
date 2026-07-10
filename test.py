from dotenv import load_dotenv
load_dotenv()
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.sammarize import generate_title,summarizes
from core.extractor import extract_action_items,extract_key_decisions,extract_questions
import os

os.getenv("SERVAM_API_KEY")
os.getcwd()


# source1="https://www.youtube.com/watch?v=7HSSR1n8dgc"
source="https://www.youtube.com/watch?v=tplWXd_T7YQ"
# source="https://www.youtube.com/watch?v_Q-e_nczWqM&t=223s"

chunks = process_input(source)

transcript=transcribe_all(chunks,True)

print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)

title = generate_title(transcript)
summary = summarizes(transcript)

print("\n" + "=" * 60)
print(f"📌 TITLE: {title}")
print("=" * 60)
print("\n📋 SUMMARY")
print("-" * 60)
print(summary)

action_items = extract_action_items(transcript)
decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)

print("\n" + "=" * 60)
print("✅ ACTION ITEMS")
print("=" * 60)
print(action_items)

print("\n" + "=" * 60)
print("🔑 KEY DECISIONS")
print("=" * 60)
print(decisions)

print("\n" + "=" * 60)
print("❓ OPEN QUESTIONS")
print("=" * 60)
print(questions)

# print(transcribe_all(chunks,translate=True))