# Customer Onboarding Process

Owner: Customer Operations · Version 3.2 · Applies to all new business customers.

## Overview
Customer onboarding turns a signed contract into an active, billable account. The target is to complete
onboarding within 5 business days of contract signature.

## Steps
1. **Contract intake (Day 0).** Sales uploads the signed contract to the CRM and sets the opportunity to "Closed Won".
   This creates an onboarding ticket automatically.
2. **Know Your Customer (KYC) checks (Day 1).** Compliance verifies the legal entity, tax ID and beneficial owners.
   Onboarding cannot continue until KYC status is "Approved".
3. **Account creation (Day 2).** Customer Operations creates the customer record in the master data system. The
   customer ID is assigned by the system and must never be typed manually.
4. **Data feed setup (Day 3).** Data Engineering adds the customer to the `customer_ingestion` pipeline so that
   orders and invoices start flowing into the lakehouse.
5. **Welcome and training (Day 4).** The account manager sends the welcome pack and schedules a training session.
6. **Go-live review (Day 5).** Customer Operations confirms the first order was processed end to end and closes the ticket.

## Escalation
If KYC is not approved within 3 business days, the account manager escalates to the Compliance lead.
Any change to the customer ID after creation requires a data-correction request to Data Engineering.
