from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel
from .minor_models import BoostedJob, Tag, Address, Category, Specialization, Website


# Enterprise model
class Enterprise(BaseModel):
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    enterpriseId: UUID
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    logoUrl: Optional[str] = None
    description: Optional[str] = None
    benefit: Optional[str] = None
    companyVision: Optional[str] = None
    foundedIn: Optional[str] = None
    organizationType: Optional[str] = None
    teamSize: Optional[str] = None
    bio: Optional[str] = None
    status: Optional[str] = None
    isPremium: Optional[bool] = None
    isTrial: Optional[bool] = None
    isActive: Optional[bool] = None
    categories: Optional[List[Category]] = []
    websites: Optional[List[Website]] = []
    addresses: Optional[List[Address]] = []


# Subset of Enterprise for Job model
class EnterpriseJob(BaseModel):
    enterpriseId: UUID
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    logoUrl: Optional[str] = None
    foundedIn: Optional[str] = None
    organizationType: Optional[str] = None
    teamSize: Optional[str] = None
    bio: Optional[str] = None
    isPremium: Optional[bool] = None
    status: Optional[str] = None
    isTrial: Optional[bool] = None


# Job model
class Job(BaseModel):
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    isActive: Optional[bool] = None
    jobId: UUID
    name: str
    lowestWage: int
    highestWage: int
    description: str
    responsibility: str
    type: str
    experience: int
    deadline: str
    introImg: Optional[str] = None
    status: str
    education: str
    isBoost: Optional[bool] = None
    enterpriseBenefits: str
    requirements: str
    tags: Optional[List[Tag]] = []
    enterprise: Optional[EnterpriseJob] = None
    addresses: Optional[List[Address]] = []
    categories: Optional[List[Category]] = []
    specializations: Optional[List[Specialization]] = []
    isFavorite: Optional[bool] = None
    applicationCount: Optional[int] = None
    boostedJob: Optional[BoostedJob] = None
