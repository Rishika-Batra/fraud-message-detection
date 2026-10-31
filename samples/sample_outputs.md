# Sample Benchmark Outputs: Fraud Message Detection

**System:** Grounded Explanation Pipeline

This document records the end-to-end pipeline execution across 5 benchmark messages, illustrating fraud detection, semantic advisory retrieval, and explanation grounding.

## SAMPLE_01: Clear Fraud (KYC / Phishing)

**Input Message:**
> Dear customer, your SBI bank account has been suspended due to pending KYC verification. Please click http://bit.ly/sbi-kyc-update to update your documents immediately.

- **Classifier Label:** `FRAUD`
- **Model Confidence:** `99.8%`
- **FAISS Similarity Scores:** `[0.5871, 0.3601]`

**Retrieved Official Sources:**
- [Fake KYC Verification and Account Blocked SMS Scam](https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372)
- [Reporting Cyber Financial Fraud and Emergency Helplines](https://cybercrime.gov.in)

**Pipeline Explanation:**

The message is fraudulent as it perfectly matches a known scam pattern [1]. It uses urgent language, claiming the SBI bank account is suspended due to pending KYC verification, a common tactic used by scammers [1]. The instruction to click a suspicious link immediately to update documents is also characteristic of fraudulent messages designed to induce panic [1]. Genuine banks do not suspend accounts without formal notice or mandate KYC updates via unsolicited SMS links [1].

**Cited Sources:**
[1] Title: Fake KYC Verification and Account Blocked SMS Scam, URL: https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372

---

## SAMPLE_02: Clear Fraud (Urgent Utility Disconnection Threat)

**Input Message:**
> URGENT: Your electricity power supply will be disconnected tonight at 9:30 PM by electricity office because your previous month bill was not updated. Please contact power officer immediately at 9876543210.

- **Classifier Label:** `FRAUD`
- **Model Confidence:** `99.9%`
- **FAISS Similarity Scores:** `[0.5656]`

**Retrieved Official Sources:**
- [Urgent Electricity Bill Disconnection Threats](TODO_VERIFY)

**Pipeline Explanation:**

This message looks fraudulent because it aligns with patterns described in advisory notes. It broadcasts an alarming threat that the electricity power supply will be cut off tonight due to an unpaid previous month's bill [1]. The message instructs the recipient to contact an unofficial personal mobile number immediately [1]. Real utility distribution companies dispatch formal written bills and never threaten instant disconnection via random mobile numbers [1].

Cited sources:
[1] Urgent Electricity Bill Disconnection Threats (TODO_VERIFY)

---

## SAMPLE_03: Borderline / Task Investment Scam

**Input Message:**
> Part-time job offer: Earn Rs 2500 - 5000 per day by simply reviewing hotels and liking YouTube videos from home. No experience needed. Contact HR manager Priya on WhatsApp now: +91-9871122334.

- **Classifier Label:** `FRAUD`
- **Model Confidence:** `99.9%`
- **FAISS Similarity Scores:** `[0.3585]`

**Retrieved Official Sources:**
- [Part-Time Job and High-Return Task-Based Investment Scams](https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372)

**Pipeline Explanation:**

This message appears fraudulent. It offers a part-time job promising high daily wages (Rs 2500-5000) for simple remote tasks like reviewing hotels and liking YouTube videos [1]. This matches the modus operandi of "Part-Time Job and High-Return Task-Based Investment Scams" where fraudsters promise high daily wages for easy tasks [1]. Additionally, recruiting via anonymous messaging like WhatsApp is not a practice of authentic organizations [1].

**Cited Sources:**
*   [1] Title: Part-Time Job and High-Return Task-Based Investment Scams, URL: https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372

---

## SAMPLE_04: Normal Legitimate (Personal Message)

**Input Message:**
> Hey, are we meeting at the campus library at 4 PM?

- **Classifier Label:** `LEGITIMATE`
- **Model Confidence:** `99.6%`
- **FAISS Similarity Scores:** `[]`

**Retrieved Official Sources:** None (Bypassed or below threshold)

**Pipeline Explanation:**

Message looks legitimate; no explanation needed.

---

## SAMPLE_05: Genuine Bank Alert / Potential Classifier False Positive

**Input Message:**
> 482910 is your secret OTP for transaction of INR 1,500.00 at Flipkart. Never share your OTP with anyone, including bank staff. If not done by you, call 1800112211 - SBI

- **Classifier Label:** `FRAUD`
- **Model Confidence:** `99.9%`
- **FAISS Similarity Scores:** `[0.4749, 0.4241, 0.38]`

**Retrieved Official Sources:**
- [One-Time Password (OTP) Confidentiality and Sharing Scams](https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372)

**Pipeline Explanation:**

The retrieved advisory notes do not indicate that this message is fraudulent. On the contrary, the message aligns with characteristics of authentic bank One-Time Passwords (OTPs) [2]. It explicitly states, "Never share your OTP with anyone, including bank staff," which is a hallmark of legitimate alerts [1], [2]. The message also accurately mentions the exact transaction amount and merchant ('INR 1,500.00 at Flipkart') [2]. It does not ask for sensitive credentials or instruct forwarding the text [1], [3]. Therefore, the classifier may be wrong, as this appears to be a genuine transaction alert.

Cited Sources:
[1] One-Time Password (OTP) Confidentiality and Sharing Scams (https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372)
[2] Characteristics of Authentic Bank One-Time Passwords (OTPs) (https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372)
[3] Impersonation of Bank Officials and Customer Support Helplines (https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372)

---

