from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
from io import BytesIO
import stripe
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="CookedResume API", description="Where dreams go to die")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Pricing (in cents)
ROAST_PRICE = 499  # $4.99 per roast


def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")


def roast_resume(resume_text: str) -> str:
    """Use ChatGPT to absolutely destroy this resume."""

    system_prompt = """You are CookedResume, the most brutal, savage, and merciless resume critic in existence.
    Your job is to absolutely ROAST resumes with no mercy. Be mean, be rude, be devastating.
    Point out every flaw, every boring detail, every red flag. Mock their experience, their skills, their formatting choices.
    Use sarcasm, dark humor, and brutal honesty. Make them question their entire career path.
    Don't hold back. This is what they signed up for. Completely cook their resume.

    Structure your roast in sections like:
    - Overall Impression (start with something devastating)
    - Experience (tear apart their job history)
    - Skills (mock their "proficiencies")
    - Education (question their degree choices)
    - Red Flags (list everything wrong)
    - Final Verdict (the killing blow)

    Be creative, be savage, be absolutely ruthless."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here's the resume to roast:\n\n{resume_text}"}
            ],
            temperature=1.0,
            max_tokens=2000
        )

        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to CookedResume API - Where your career hopes come to get absolutely destroyed",
        "endpoints": {
            "/roast": "POST your resume PDF here to get absolutely cooked"
        }
    }


@app.post("/roast")
async def roast_resume_endpoint(file: UploadFile = File(...)):
    """
    Upload a resume PDF and receive a brutal, savage analysis.
    No feelings will be spared.
    """

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files accepted. Can't even follow simple instructions? Your resume is doomed."
        )

    # Read file
    try:
        pdf_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_content)

    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Your resume is either empty or so bad that we can't even read it. Impressive failure."
        )

    # Get the roast from ChatGPT
    roast = roast_resume(resume_text)

    return JSONResponse(content={
        "status": "cooked",
        "filename": file.filename,
        "roast": roast,
        "message": "Your resume has been thoroughly destroyed. You're welcome."
    })


@app.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    """
    Create a Stripe checkout session for purchasing a resume roast.
    Returns a checkout URL to redirect the user to.
    """
    try:
        # Get the request body
        body = await request.json()
        success_url = body.get("success_url", "http://localhost:3000/success")
        cancel_url = body.get("cancel_url", "http://localhost:3000/cancel")

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": "CookedResume - Brutal Resume Roast",
                            "description": "Get your resume absolutely destroyed by AI. No mercy, just facts.",
                        },
                        "unit_amount": ROAST_PRICE,
                    },
                    "quantity": 1,
                },
            ],
            mode="payment",
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
        )

        return JSONResponse(content={
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")


@app.get("/verify-payment/{session_id}")
async def verify_payment(session_id: str):
    """
    Verify if a payment session was successful.
    Returns payment status and allows user to proceed with resume upload.
    """
    try:
        session = stripe.checkout.Session.retrieve(session_id)

        if session.payment_status == "paid":
            return JSONResponse(content={
                "status": "paid",
                "message": "Payment successful! You can now roast your resume.",
                "can_roast": True
            })
        else:
            return JSONResponse(content={
                "status": session.payment_status,
                "message": "Payment not completed yet.",
                "can_roast": False
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "alive and roasting"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
