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
from clerk_backend_api import Clerk
import json
from datetime import datetime
from pathlib import Path

load_dotenv()

app = FastAPI(title="CookedResume API", description="Where dreams go to die")

# CORS middleware - allow both local and production frontend
allowed_origins = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "https://cookedresume.com",
    "https://www.cookedresume.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Initialize Clerk
clerk_client = Clerk(bearer_auth=os.getenv("CLERK_SECRET_KEY"))

# Pricing (in cents)
ROAST_PRICE = 100  # $1.00 per roast

# Analytics file path
UPLOADS_LOG_FILE = Path(__file__).parent / "resume_uploads.json"


def log_resume_upload(filename: str, file_size: int, success: bool = True, error: str = None):
    """
    Log resume upload event to JSON file for analytics.

    Args:
        filename: Name of the uploaded file
        file_size: Size of the file in bytes
        success: Whether the upload was successful
        error: Error message if upload failed
    """
    upload_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "filename": filename,
        "file_size_bytes": file_size,
        "success": success,
        "error": error
    }

    try:
        # Read existing data or create new list
        if UPLOADS_LOG_FILE.exists():
            with open(UPLOADS_LOG_FILE, 'r') as f:
                uploads = json.load(f)
        else:
            uploads = []

        # Append new upload
        uploads.append(upload_data)

        # Write back to file
        with open(UPLOADS_LOG_FILE, 'w') as f:
            json.dump(uploads, f, indent=2)
    except Exception as e:
        # Don't fail the request if logging fails
        print(f"Failed to log upload: {str(e)}")


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


def parse_resume_sections(resume_text: str) -> dict:
    """Use ChatGPT to parse and identify resume sections with their text content."""

    system_prompt = """You are a resume parser. Analyze the resume text and identify distinct sections.
    Return a JSON object where each section contains its extracted text content.

    Format:
    {
        "contact_info": {
            "content": "the actual text from this section"
        },
        "summary": {
            "content": "the actual summary text"
        },
        "experience": {
            "content": "all work experience text"
        },
        "education": {
            "content": "all education text"
        },
        "skills": {
            "content": "all skills text"
        }
    }

    Common sections to look for:
    - contact_info (name, email, phone, location, linkedin)
    - summary or objective
    - experience or work_experience
    - education
    - skills
    - projects
    - certifications
    - awards

    Use lowercase with underscores for section names. Include ONLY sections that exist in the resume.
    Each section must have a "content" field with the actual text from that section.
    Return ONLY valid JSON, no additional text or markdown."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Parse this resume into sections:\n\n{resume_text}"}
            ],
            temperature=0.3,
            max_tokens=3000
        )

        # Parse the JSON response
        import json
        sections = json.loads(response.choices[0].message.content)
        return sections
    except Exception as e:
        # If parsing fails, return the whole text as one section
        return {"full_resume": {"content": resume_text}}


def roast_resume(resume_text: str) -> str:
    """Use ChatGPT to provide brutally honest resume feedback."""

    system_prompt = """You are CookedResume, a brutally honest resume reviewer who tells it like it is.
    Your job is to provide direct, no-nonsense feedback that actually helps people improve their resumes.

    Be blunt and straightforward - don't sugarcoat problems, but provide actionable advice.
    Point out what's wrong and explain WHY it's wrong, then tell them HOW to fix it.
    If something is good, acknowledge it. If it's bad, say so clearly and offer specific improvements.

    Focus on:
    - What's actually hurting their chances of getting interviews
    - Concrete improvements they can make immediately
    - Industry standards they're not meeting
    - Missing information that recruiters need to see
    - Poor formatting or structure issues

    Structure your review in clear sections:
    - Overall Impression (be direct about first impressions)
    - Experience Section (what's working, what's not, specific fixes)
    - Skills Section (relevance, presentation, gaps)
    - Education & Certifications (if applicable)
    - Formatting & Structure (readability, ATS compatibility)
    - Key Action Items (prioritized list of improvements)

    Be honest, be direct, be helpful. No fluff, no false encouragement, just practical advice that works."""

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
        file_size = len(pdf_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Extract text from PDF
    try:
        resume_text = extract_text_from_pdf(pdf_content)

        if not resume_text or len(resume_text.strip()) < 50:
            log_resume_upload(file.filename, file_size, success=False, error="Empty or unreadable PDF")
            raise HTTPException(
                status_code=400,
                detail="Your resume is either empty or so bad that we can't even read it. Impressive failure."
            )

        # Parse resume into sections
        resume_sections = parse_resume_sections(resume_text)

        # Get the roast from ChatGPT
        roast = roast_resume(resume_text)

        # Log successful upload
        log_resume_upload(file.filename, file_size, success=True)

        return JSONResponse(content={
            "status": "cooked",
            "filename": file.filename,
            "resume_text": resume_text,
            "resume_sections": resume_sections,
            "roast": roast,
            "message": "Your resume has been thoroughly destroyed. You're welcome."
        })
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log failed upload
        log_resume_upload(file.filename, file_size, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


@app.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    """
    Create a Stripe checkout session for purchasing a resume roast.
    Returns a checkout URL to redirect the user to.
    """
    try:
        # Get the request body
        body = await request.json()
        success_url = body.get("success_url", "http://localhost:8080/success")
        cancel_url = body.get("cancel_url", "http://localhost:8080/cancel")
        clerk_user_id = body.get("clerk_user_id")

        # Build metadata only if clerk_user_id is provided
        metadata = {}
        if clerk_user_id:
            metadata["clerk_user_id"] = clerk_user_id

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
            metadata=metadata
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


@app.post("/link-payment")
async def link_payment(request: Request):
    """
    Link a Stripe payment to a Clerk user account.
    Called when user signs up/in after making a payment while not authenticated.
    """
    try:
        body = await request.json()
        stripe_session_id = body.get("stripe_session_id")
        clerk_user_id = body.get("clerk_user_id")

        if not stripe_session_id or not clerk_user_id:
            raise HTTPException(
                status_code=400,
                detail="stripe_session_id and clerk_user_id are required"
            )

        # Verify payment session exists and was paid
        session = stripe.checkout.Session.retrieve(stripe_session_id)

        if session.payment_status != "paid":
            raise HTTPException(
                status_code=400,
                detail="Payment not completed"
            )

        # Update Clerk user metadata to unlock report
        clerk_client.users.update(
            user_id=clerk_user_id,
            public_metadata={
                "hasUnlockedReport": True
            }
        )

        return JSONResponse(content={
            "status": "success",
            "message": "Payment successfully linked to your account"
        })

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook endpoint to handle payment events.
    Handles: checkout.session.completed, checkout.session.async_payment_succeeded,
             checkout.session.async_payment_failed, checkout.session.expired
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Get event type and data
    event_type = event["type"]
    session = event["data"]["object"]

    # Handle checkout session completed
    if event_type == "checkout.session.completed":
        print(f"✅ Payment successful for session: {session['id']}")

        # Get clerk_user_id from metadata
        clerk_user_id = session.get("metadata", {}).get("clerk_user_id")

        if clerk_user_id:
            try:
                # Update Clerk user metadata to unlock report
                clerk_client.users.update(
                    user_id=clerk_user_id,
                    public_metadata={
                        "hasUnlockedReport": True
                    }
                )
                print(f"✅ Updated Clerk user {clerk_user_id}: hasUnlockedReport = True")
            except Exception as e:
                print(f"❌ Failed to update Clerk user {clerk_user_id}: {str(e)}")

    # Handle async payment succeeded
    elif event_type == "checkout.session.async_payment_succeeded":
        print(f"✅ Async payment succeeded for session: {session['id']}")

        # Get clerk_user_id from metadata
        clerk_user_id = session.get("metadata", {}).get("clerk_user_id")

        if clerk_user_id:
            try:
                # Update Clerk user metadata to unlock report
                clerk_client.users.update(
                    user_id=clerk_user_id,
                    public_metadata={
                        "hasUnlockedReport": True
                    }
                )
                print(f"✅ Updated Clerk user {clerk_user_id}: hasUnlockedReport = True")
            except Exception as e:
                print(f"❌ Failed to update Clerk user {clerk_user_id}: {str(e)}")

    # Handle async payment failed
    elif event_type == "checkout.session.async_payment_failed":
        print(f"❌ Async payment failed for session: {session['id']}")

    # Handle session expired
    elif event_type == "checkout.session.expired":
        print(f"⏱️ Session expired: {session['id']}")

    return JSONResponse(content={"status": "success"})


@app.get("/health")
def health_check():
    return {"status": "alive and roasting"}


@app.get("/analytics/uploads")
def get_upload_analytics():
    """
    Get analytics about resume uploads.
    Returns total count, success/failure breakdown, and recent uploads.
    """
    try:
        if not UPLOADS_LOG_FILE.exists():
            return JSONResponse(content={
                "total_uploads": 0,
                "successful_uploads": 0,
                "failed_uploads": 0,
                "recent_uploads": []
            })

        with open(UPLOADS_LOG_FILE, 'r') as f:
            uploads = json.load(f)

        total = len(uploads)
        successful = sum(1 for u in uploads if u.get("success", False))
        failed = total - successful

        # Get last 10 uploads
        recent = uploads[-10:] if len(uploads) > 10 else uploads
        recent.reverse()  # Most recent first

        return JSONResponse(content={
            "total_uploads": total,
            "successful_uploads": successful,
            "failed_uploads": failed,
            "success_rate": f"{(successful/total*100):.1f}%" if total > 0 else "0%",
            "recent_uploads": recent
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading analytics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
