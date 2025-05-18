from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel


# Tag model
class Tag(BaseModel):
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    isActive: Optional[bool]
    tagId: UUID
    name: str
    color: str
    backgroundColor: Optional[str]


# Address model
class Address(BaseModel):
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    isActive: bool
    addressId: UUID
    country: str
    city: str
    street: str
    zipCode: str
    mixedAddress: str


# Category model
class Category(BaseModel):
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    isActive: Optional[bool]
    categoryId: UUID
    categoryName: str


# Specialization model
class Specialization(Category):
    pass


# Website model


class Website(BaseModel):
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    isActive: Optional[bool] = None
    websiteId: UUID
    socialType: str
    socialLink: str


class BoostedJob(BaseModel):
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    isActive: Optional[bool] = None
    id: UUID
    pointsUsed: int
