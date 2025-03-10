from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompts import PromptTemplate 
from langchain.schema import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from enum import Enum

GROQ_API_KEY = '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'

CustomerDataTemplate = """
    Based on the following query from the User
    query: {query}
    Target is to transform this query to JSON with a specific format
    Please keep this Order
        gender
        SeniorCitizen
        Partner
        Dependents
        tenure
        PhoneService
        MultipleLines
        InternetService
        OnlineSecurity
        OnlineBackup
        DeviceProtection
        TechSupport
        StreamingTV
        StreamingMovies
        Contract
        PaperlessBilling
        PaymentMethod
        MonthlyCharges
        TotalCharges
    """

class Gender(str, Enum):
    male = 'Male'
    female = 'Female'

class YesNo(str, Enum):
    yes = 'Yes'
    no = 'No'

class MultipleLine(str, Enum):
    yes = 'Yes'
    no = 'No'
    no_phone_service = 'No phone service'

class InternetServices(str, Enum):
    dsl = 'DSL'
    fiber_optic = 'Fiber optic'
    no = 'No'

class InternetServiceRelated(str, Enum):
    yes = 'Yes'
    no = 'No'
    no_internet_service = 'No internet service'

class Contracts(str, Enum):
    month_to_month = 'Month-to-month'
    one_year = 'One year'
    two_year = 'Two year'

class PaymentMethods(str, Enum):
    electronic_check = 'Electronic check'
    mailed_check = 'Mailed check'
    bank_transfer = 'Bank transfer (automatic)'
    credit_card = 'Credit card (automatic)'

class LLMCustomerData(BaseModel):
    gender: Gender = Field(..., description="The gender of the customer: Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen: 0 or 1")
    Partner: YesNo = Field(..., description="Whether the customer has a partner: Yes or No")
    Dependents: YesNo = Field(..., description="Whether the customer has dependents: Yes or No")
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company")
    PhoneService: YesNo = Field(..., description="Whether the customer has phone service: Yes or No")
    MultipleLines: MultipleLine = Field(..., description="Whether the customer has multiple lines: Yes, No, or No phone service")
    InternetService: InternetServices = Field(..., description="The type of internet service the customer has: DSL, Fiber optic, or No")
    OnlineSecurity: InternetServiceRelated = Field(..., description="Whether the customer has online security: Yes, No, or No internet service")
    OnlineBackup: InternetServiceRelated = Field(..., description="Whether the customer has online backup: Yes, No, or No internet service")
    DeviceProtection: InternetServiceRelated = Field(..., description="Whether the customer has device protection: Yes, No, or No internet service")
    TechSupport: InternetServiceRelated = Field(..., description="Whether the customer has tech support: Yes, No, or No internet service")
    StreamingTV: InternetServiceRelated = Field(..., description="Whether the customer has streaming TV: Yes, No, or No internet service")
    StreamingMovies: InternetServiceRelated = Field(..., description="Whether the customer has streaming movies: Yes, No, or No internet service")
    Contract: Contracts = Field(..., description="The contract term of the customer: Month-to-month, One year, or Two year")
    PaperlessBilling: YesNo = Field(..., description="Whether the customer has paperless billing: Yes or No")
    PaymentMethod: PaymentMethods = Field(..., description="The payment method used by the customer: Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0, description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., ge=0, description="The total amount charged to the customer")


def convert_natural_language_to_json(llm, query):
    content = PromptTemplate.from_template(CustomerDataTemplate).format(query=query)
    dict_schema = convert_to_openai_tool(LLMCustomerData)
    structured_llm = llm.with_structured_output(dict_schema)
    response = structured_llm.invoke([HumanMessage(content=content)])
    return response

